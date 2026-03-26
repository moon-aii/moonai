#include "simulation/session.hpp"

#include "core/profiler_macros.hpp"
#include "data/logger.hpp"
#include "evolution/evolution_manager.hpp"
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"
#include "simulation/components.hpp"
#include "simulation/registry.hpp"
#include "simulation/simulation_manager.hpp"
#include "visualization/visualization_manager.hpp"
#include <csignal>

#include <spdlog/spdlog.h>

#include <chrono>
#include <limits>
#include <memory>

namespace moonai {

// Static signal handling members
volatile sig_atomic_t Session::g_running_ = 1;
bool Session::handlers_registered_ = false;

void Session::signal_handler(int) {
  g_running_ = 0;
}

void Session::register_signal_handlers() {
  if (!handlers_registered_) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    handlers_registered_ = true;
  }
  g_running_ = 1; // Reset for this session
}

void Session::restore_signal_handlers() {
  // Note: We don't actually restore old handlers for simplicity
  // The handlers remain registered but g_running_ is reset for new sessions
}

Session::Session(const SessionConfig &cfg)
    : cfg_(cfg),
      rng_(
          cfg.seed == 0
              ? static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count())
              : cfg.seed),
      simulation_(cfg.sim_config), evolution_(cfg.sim_config, rng_),
      logger_(cfg.output_dir, cfg.seed,
              cfg.run_name_override.value_or(cfg.experiment_name)),
      steps_executed_(0), births_in_window_(0), deaths_in_window_(0) {
  // Validate config
  const auto errors = validate_config(cfg_.sim_config);
  if (!errors.empty()) {
    for (const auto &error : errors) {
      spdlog::error("Config error [{}]: {}", error.field, error.message);
    }
    throw std::runtime_error("Invalid simulation configuration");
  }

  // Initialize simulation and evolution
  simulation_.initialize();
  evolution_.initialize(SensorSoA::INPUT_COUNT, 2);
  evolution_.seed_initial_population_ecs(registry_);
  simulation_.refresh_state_ecs(registry_);

  // Initialize logger
  logger_.initialize(cfg_.sim_config);

  // Initialize visualization if not headless
  if (!cfg_.headless) {
    visualization_ = std::make_unique<VisualizationManager>(cfg_.sim_config);
    if (!visualization_->initialize()) {
      spdlog::error("Failed to initialize visualization");
      visualization_.reset();
    }
  }

  // Enable GPU if available
  evolution_.enable_gpu(cfg_.enable_gpu);

  // Register signal handlers
  register_signal_handlers();
}

Session::~Session() {
  restore_signal_handlers();
}

Registry &Session::registry() {
  return registry_;
}
SimulationManager &Session::simulation() {
  return simulation_;
}
EvolutionManager &Session::evolution() {
  return evolution_;
}
MetricsCollector &Session::metrics() {
  return metrics_;
}
Logger &Session::logger() {
  return logger_;
}
VisualizationManager *Session::visualization() {
  return visualization_.get();
}

const Registry &Session::registry() const {
  return registry_;
}
const SimulationManager &Session::simulation() const {
  return simulation_;
}
const EvolutionManager &Session::evolution() const {
  return evolution_;
}
const MetricsCollector &Session::metrics() const {
  return metrics_;
}

int Session::steps_executed() const {
  return steps_executed_;
}
int Session::births_in_window() const {
  return births_in_window_;
}
int Session::deaths_in_window() const {
  return deaths_in_window_;
}

void Session::record_birth() {
  ++births_in_window_;
}
void Session::record_death() {
  ++deaths_in_window_;
}
void Session::reset_window_counters() {
  births_in_window_ = 0;
  deaths_in_window_ = 0;
}

void Session::step(float dt) {
  // Compute actions
  actions_buffer_.clear();
  evolution_.compute_actions_ecs(registry_, actions_buffer_);

  // Apply actions to entities
  MOONAI_PROFILE_SCOPE("evolution_apply_actions");
  size_t action_idx = 0;
  for (Entity e : registry_.living_entities()) {
    size_t idx = registry_.index_of(e);
    if (!registry_.vitals().alive[idx]) {
      continue;
    }

    if (action_idx < actions_buffer_.size()) {
      float dx = actions_buffer_[action_idx].x;
      float dy = actions_buffer_[action_idx].y;
      float speed = registry_.motion().speed[idx];

      registry_.motion().vel_x[idx] = dx * speed;
      registry_.motion().vel_y[idx] = dy * speed;
      registry_.positions().x[idx] += registry_.motion().vel_x[idx] * dt;
      registry_.positions().y[idx] += registry_.motion().vel_y[idx] * dt;
      registry_.stats().distance_traveled[idx] +=
          std::sqrt(dx * dx + dy * dy) * speed * dt;

      ++action_idx;
    }
  }

  // Step simulation
  simulation_.step_ecs(registry_, dt);

  // Handle reproduction
  auto pairs = simulation_.find_reproduction_pairs_ecs(registry_);
  for (const auto &pair : pairs) {
    Entity child = evolution_.create_offspring_ecs(
        registry_, pair.parent_a, pair.parent_b, pair.spawn_position);
    if (child != INVALID_ENTITY) {
      ++births_in_window_;
      // Record birth event
      simulation_.record_event(SimEvent{SimEvent::Birth, child, child,
                                        pair.parent_a, pair.parent_b,
                                        pair.spawn_position});
    }
  }

  // Refresh fitness
  evolution_.refresh_fitness_ecs(registry_);

  // Update species periodically
  if (cfg_.sim_config.species_update_interval_steps > 0 &&
      (steps_executed_ % cfg_.sim_config.species_update_interval_steps) == 0) {
    evolution_.refresh_species_ecs(registry_);
  }

  // Record deaths
  for (const auto &event : simulation_.last_events()) {
    if (event.type == SimEvent::Death) {
      ++deaths_in_window_;
    }
  }

  ++steps_executed_;
}

StepMetrics Session::record_and_log(int births, int deaths) {
  evolution_.refresh_species_ecs(registry_);

  float best_predator = 0.0f, avg_predator = 0.0f;
  float best_prey = 0.0f, avg_prey = 0.0f;
  evolution_.get_fitness_by_type_ecs(registry_, best_predator, avg_predator,
                                     best_prey, avg_prey);

  int alive_predators = 0, alive_prey = 0;
  for (Entity e : registry_.living_entities()) {
    size_t idx = registry_.index_of(e);
    if (registry_.vitals().alive[idx]) {
      if (registry_.identity().type[idx] == IdentitySoA::TYPE_PREDATOR) {
        ++alive_predators;
      } else {
        ++alive_prey;
      }
    }
  }

  auto snapshot =
      metrics_.collect_ecs(steps_executed_, registry_, evolution_, births,
                           deaths, evolution_.species_count());

  logger_.log_report(snapshot);

  // Find and log best genome
  const Genome *best_genome = nullptr;
  float best_fitness = -std::numeric_limits<float>::infinity();
  for (Entity e : registry_.living_entities()) {
    const auto *genome = evolution_.genome_for(e);
    if (genome && genome->fitness() > best_fitness) {
      best_fitness = genome->fitness();
      best_genome = genome;
    }
  }
  if (best_genome) {
    logger_.log_best_genome(steps_executed_, *best_genome);
  }

  logger_.log_species(steps_executed_, evolution_.species());
  logger_.flush();

  reset_window_counters();
  return snapshot;
}

void Session::update_selected_visualization() {
  if (!visualization_ || !cfg_.enable_interactions) {
    return;
  }

  Entity selected = visualization_->selected_entity();
  if (selected == INVALID_ENTITY || !registry_.valid(selected)) {
    return;
  }

  size_t idx = registry_.index_of(selected);
  const auto *genome = evolution_.genome_for(selected);
  if (!genome || !registry_.vitals().alive[idx]) {
    return;
  }

  const float *sensors = registry_.sensors().input_ptr(idx);
  std::vector<float> sensor_vec(sensors, sensors + SensorSoA::INPUT_COUNT);

  NeuralNetwork *network = evolution_.network_cache().get_network(selected);
  if (network) {
    network->activate(sensor_vec);
    visualization_->set_selected_activations(network->last_activations(),
                                             network->node_index_map());
  }
}

bool Session::should_continue() const {
  // Check max steps
  if (cfg_.sim_config.max_steps > 0 &&
      steps_executed_ >= cfg_.sim_config.max_steps) {
    return false;
  }

  // Check signal
  if (g_running_ == 0) {
    return false;
  }

  return true;
}

void Session::log_report(const StepMetrics &snapshot) const {
  spdlog::info(
      "Step {:6d}: predators={} prey={} births={} deaths={} species={}",
      snapshot.step, snapshot.predator_count, snapshot.prey_count,
      snapshot.births, snapshot.deaths, snapshot.num_species);
}

void Session::log_stop_reason(StopReason reason) const {
  switch (reason) {
    case StopReason::UserQuit:
      spdlog::info("Simulation stopped by user (window closed)");
      break;
    case StopReason::Signal:
      spdlog::info("Simulation stopped by signal (Ctrl+C)");
      break;
    case StopReason::Completed:
      // Don't log for normal completion
      break;
  }
}

StopReason Session::run() {
  const float dt = 1.0f / static_cast<float>(cfg_.sim_config.target_fps);
  StopReason stop_reason = StopReason::Completed;

  if (cfg_.headless) {
    // Headless mode: run as fast as possible
    while (should_continue()) {
      step(dt);

      // Check for report interval
      if (steps_executed_ % cfg_.sim_config.report_interval_steps == 0) {
        auto snapshot = record_and_log(births_in_window_, deaths_in_window_);
        log_report(snapshot);

        // Call custom callback if provided
        if (cfg_.on_report_callback) {
          cfg_.on_report_callback(snapshot);
        }
      }
    }

    // Final record if needed
    if (births_in_window_ > 0 || deaths_in_window_ > 0 ||
        metrics_.history().empty()) {
      record_and_log(births_in_window_, deaths_in_window_);
    }

    logger_.flush();

    stop_reason =
        (g_running_ == 0) ? StopReason::Signal : StopReason::Completed;
  } else {
    // GUI mode
    if (!visualization_) {
      spdlog::error("Visualization requested but not initialized");
      return StopReason::Completed;
    }

    while (should_continue()) {
      MOONAI_PROFILE_SCOPE("frame_total");

      // Handle events
      {
        MOONAI_PROFILE_SCOPE("handle_events");
        visualization_->handle_events();
      }

      // Check for window close
      if (visualization_->should_close()) {
        stop_reason = StopReason::UserQuit;
        break;
      }

      // Check signal during loop
      if (g_running_ == 0) {
        stop_reason = StopReason::Signal;
        break;
      }

      // Handle pause (unless auto_run)
      if (!cfg_.auto_run && visualization_->is_paused() &&
          !visualization_->should_step()) {
        MOONAI_PROFILE_SCOPE("render");
        visualization_->render_ecs(registry_, evolution_, simulation_,
                                   steps_executed_);
        continue;
      }

      if (!cfg_.auto_run) {
        visualization_->clear_step();
      }

      // Run simulation steps
      int steps_to_run = cfg_.auto_run ? cfg_.speed_multiplier
                                       : visualization_->speed_multiplier();
      steps_to_run = std::max(1, steps_to_run);

      for (int i = 0; i < steps_to_run && should_continue(); ++i) {
        step(dt);

        // Check for report interval
        if (steps_executed_ % cfg_.sim_config.report_interval_steps == 0) {
          auto snapshot = record_and_log(births_in_window_, deaths_in_window_);
          log_report(snapshot);

          // Call custom callback if provided
          if (cfg_.on_report_callback) {
            cfg_.on_report_callback(snapshot);
          }
        }
      }

      // Update selected visualization (only if interactions enabled)
      if (cfg_.enable_interactions) {
        MOONAI_PROFILE_SCOPE("update_selected");
        update_selected_visualization();
      }

      // Render
      {
        MOONAI_PROFILE_SCOPE("render");
        visualization_->render_ecs(registry_, evolution_, simulation_,
                                   steps_executed_);
      }
    }

    // Final record if needed
    if (births_in_window_ > 0 || deaths_in_window_ > 0 ||
        metrics_.history().empty()) {
      record_and_log(births_in_window_, deaths_in_window_);
    }

    logger_.flush();
  }

  // Log stop reason
  log_stop_reason(stop_reason);

  return stop_reason;
}

} // namespace moonai
