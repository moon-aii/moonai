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
#include <cstdlib>

#include <spdlog/spdlog.h>

#include <chrono>
#include <limits>
#include <memory>

namespace moonai {

// Static signal handling members
volatile sig_atomic_t Session::g_running_ = 1;

void Session::signal_handler(int) {
  g_running_ = 0;
}

void Session::register_signal_handlers() {
  static bool registered = false;
  if (!registered) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    registered = true;
  }
  g_running_ = 1; // Reset for this session
}

Session::Session(const SessionConfig &cfg)
    : cfg_(cfg),
      rng_(
          cfg.sim_config.seed == 0
              ? static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count())
              : cfg.sim_config.seed),
      simulation_(cfg.sim_config), evolution_(cfg.sim_config, rng_),
      logger_(cfg.sim_config.output_dir, cfg.sim_config.seed,
              cfg.run_name_override.value_or(cfg.experiment_name)),
      steps_executed_(0) {
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
  if (cfg_.enable_gpu) {
    evolution_.enable_gpu(true);
    simulation_.enable_gpu(true);
    spdlog::info("GPU acceleration enabled");
  }

  // Register signal handlers
  register_signal_handlers();
}

void Session::step() {
  MOONAI_PROFILE_SCOPE("session_step");

  if (cfg_.enable_gpu && simulation_.gpu_enabled()) {
    simulation_.step_gpu_ecs(registry_, evolution_);
  } else {
    simulation_.step_ecs(registry_, evolution_);
  }

  // Handle reproduction
  auto pairs = simulation_.find_reproduction_pairs_ecs(registry_);
  for (const auto &pair : pairs) {
    Entity child = evolution_.create_offspring_ecs(
        registry_, pair.parent_a, pair.parent_b, pair.spawn_position);
    if (child != INVALID_ENTITY) {
      // Record birth event (births counted from events in MetricsCollector)
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

  // Note: Deaths are recorded by SimulationManager and counted from events
  // in MetricsCollector

  ++steps_executed_;
}

StepMetrics Session::record_and_log() {
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

  auto snapshot = metrics_.collect_ecs(steps_executed_, registry_, evolution_,
                                       simulation_, evolution_.species_count());

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

  return snapshot;
}

void Session::update_selected_visualization() {
  if (!visualization_ || !cfg_.interactive) {
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

void Session::log_early_stop(bool user_quit) const {
  if (user_quit) {
    spdlog::info("Simulation stopped by user (window closed)");
  } else {
    spdlog::info("Simulation stopped by signal (Ctrl+C)");
  }
}

bool Session::run() {
  // Check for display availability if GUI mode requested
  if (!cfg_.headless) {
    if (std::getenv("DISPLAY") == nullptr &&
        std::getenv("WAYLAND_DISPLAY") == nullptr) {
      spdlog::error("No display server found. GUI mode requires a display.");
      return false;
    }
  }

  bool completed = true;
  bool user_quit = false;

  if (cfg_.headless) {
    // Headless mode: run as fast as possible
    while (should_continue()) {
      step();

      // Check for report interval
      if (steps_executed_ % cfg_.sim_config.report_interval_steps == 0) {
        auto snapshot = record_and_log();
        log_report(snapshot);
      }
    }

    // Final record if needed
    if (metrics_.history().empty()) {
      record_and_log();
    }

    logger_.flush();

    completed = (g_running_ != 0);
  } else {
    // GUI mode
    if (!visualization_) {
      spdlog::error("Visualization requested but not initialized");
      return true;
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
        completed = false;
        user_quit = true;
        break;
      }

      // Check signal during loop
      if (g_running_ == 0) {
        completed = false;
        break;
      }

      const bool step_requested =
          cfg_.interactive && visualization_->should_step();

      // Handle pause (when interactive)
      if (cfg_.interactive && visualization_->is_paused() && !step_requested) {
        MOONAI_PROFILE_SCOPE("render");
        visualization_->render_ecs(registry_, evolution_, simulation_,
                                   steps_executed_);
        continue;
      }

      if (cfg_.interactive && step_requested) {
        visualization_->clear_step();
      }

      // Run simulation steps
      int steps_to_run =
          step_requested
              ? 1
              : (cfg_.interactive ? visualization_->speed_multiplier()
                                  : cfg_.speed_multiplier);
      steps_to_run = std::max(1, steps_to_run);

      for (int i = 0; i < steps_to_run && should_continue(); ++i) {
        step();

        // Check for report interval
        if (steps_executed_ % cfg_.sim_config.report_interval_steps == 0) {
          auto snapshot = record_and_log();
          log_report(snapshot);
        }
      }

      // Update selected visualization (only if interactive)
      if (cfg_.interactive) {
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
    if (metrics_.history().empty()) {
      record_and_log();
    }

    logger_.flush();
  }

  // Log early stop
  if (!completed) {
    log_early_stop(user_quit);
  }

  spdlog::info("Output saved to: {}", logger_.run_dir());

  return completed;
}

} // namespace moonai
