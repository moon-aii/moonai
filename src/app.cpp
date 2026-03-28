#include "app.hpp"

#include "core/profiler_macros.hpp"
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"
#include "simulation/components.hpp"
#include "visualization/visual_constants.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace moonai {

namespace {

int count_active_food(const FoodStore &food_store) {
  int active_food = 0;
  for (uint8_t active : food_store.active()) {
    active_food += active ? 1 : 0;
  }
  return active_food;
}

Vec2 wrap_diff(Vec2 diff, float world_width, float world_height) {
  if (std::abs(diff.x) > world_width * 0.5f) {
    diff.x = diff.x > 0.0f ? diff.x - world_width : diff.x + world_width;
  }
  if (std::abs(diff.y) > world_height * 0.5f) {
    diff.y = diff.y > 0.0f ? diff.y - world_height : diff.y + world_height;
  }
  return diff;
}

} // namespace

volatile std::sig_atomic_t App::g_running_ = 1;

void App::signal_handler(int) {
  g_running_ = 0;
}

void App::register_signal_handlers() {
  static bool registered = false;
  if (!registered) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    registered = true;
  }
  g_running_ = 1;
}

App::App(const AppConfig &cfg)
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
  const auto errors = validate_config(cfg_.sim_config);
  if (!errors.empty()) {
    for (const auto &error : errors) {
      spdlog::error("Config error [{}]: {}", error.field, error.message);
    }
    throw std::runtime_error("Invalid simulation configuration");
  }

  simulation_.initialize();
  evolution_.initialize(SensorSoA::INPUT_COUNT, 2);
  evolution_.seed_initial_population(registry_);
  simulation_.refresh_state(registry_);

  logger_.initialize(cfg_.sim_config);

  if (!cfg_.headless) {
    visualization_ = std::make_unique<VisualizationManager>(cfg_.sim_config);
    if (!visualization_->initialize()) {
      spdlog::error("Failed to initialize visualization");
      visualization_.reset();
    }
  }

  if (cfg_.enable_gpu) {
    evolution_.enable_gpu(true);
    simulation_.enable_gpu(true);
    spdlog::info("GPU acceleration enabled");
  }

  register_signal_handlers();
}

void App::step() {
  MOONAI_PROFILE_SCOPE("step");

  SimulationManager::SimulationStepResult step_result;
  if (cfg_.enable_gpu && simulation_.gpu_enabled()) {
    step_result = simulation_.step_gpu(registry_, evolution_);
  } else {
    step_result = simulation_.step(registry_, evolution_);
  }

  last_step_events_ = std::move(step_result.events);

  for (const auto &pair : step_result.reproduction_pairs) {
    Entity child = evolution_.create_offspring(
        registry_, pair.parent_a, pair.parent_b, pair.spawn_position);
    if (child != INVALID_ENTITY) {
      last_step_events_.push_back(
          SimEvent{SimEvent::Birth, child, child, pair.spawn_position});
    }
  }

  accumulate_events(last_step_events_);

  if (cfg_.sim_config.species_update_interval_steps > 0 &&
      (steps_executed_ % cfg_.sim_config.species_update_interval_steps) == 0) {
    evolution_.refresh_species(registry_);
  }

  ++steps_executed_;
}

StepMetrics App::record_and_log() {
  evolution_.refresh_species(registry_);

  auto snapshot =
      metrics_.collect(steps_executed_, registry_, evolution_,
                       last_step_events_, evolution_.species_count());

  logger_.log_report(snapshot);

  const Genome *best_genome = nullptr;
  int best_complexity = -1;
  for (Entity e : registry_.living_entities()) {
    const auto *genome = evolution_.genome_for(e);
    if (genome && genome->complexity() > best_complexity) {
      best_complexity = genome->complexity();
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

void App::update_selected_visualization() {
  selected_node_activations_.clear();

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
    for (const auto &[node_id, node_index] : network->node_index_map()) {
      if (node_index >= 0 &&
          node_index < static_cast<int>(network->last_activations().size())) {
        selected_node_activations_[node_id] =
            network->last_activations()[node_index];
      }
    }
  }
}

FrameSnapshot App::build_frame_snapshot() const {
  FrameSnapshot frame;
  if (!visualization_) {
    return frame;
  }

  frame.world_width = cfg_.sim_config.grid_size;
  frame.world_height = cfg_.sim_config.grid_size;
  const int active_food = count_active_food(simulation_.food_store());

  float pred_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float prey_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int alive_predators = 0;
  int alive_prey = 0;

  const auto &living = registry_.living_entities();
  const auto &positions = registry_.positions();
  const auto &motion = registry_.motion();
  const auto &identity = registry_.identity();
  const auto &vitals = registry_.vitals();
  const auto &stats = registry_.stats();

  frame.foods.reserve(static_cast<std::size_t>(active_food));
  for (std::size_t i = 0; i < simulation_.food_store().size(); ++i) {
    if (!simulation_.food_store().active()[i]) {
      continue;
    }
    frame.foods.push_back(
        RenderFood{Vec2{simulation_.food_store().pos_x()[i],
                        simulation_.food_store().pos_y()[i]}});
  }

  for (Entity entity : living) {
    size_t idx = registry_.index_of(entity);
    if (!vitals.alive[idx]) {
      continue;
    }

    float energy_ratio = vitals.energy[idx] / cfg_.sim_config.initial_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);

    if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
      ++alive_predators;
      pred_dist[bucket] += 1.0f;
    } else {
      ++alive_prey;
      prey_dist[bucket] += 1.0f;
    }

    frame.agents.push_back(RenderAgent{
        entity, Vec2{positions.x[idx], positions.y[idx]},
        Vec2{motion.vel_x[idx], motion.vel_y[idx]}, identity.type[idx]});
  }

  if (alive_predators > 0) {
    for (float &value : pred_dist) {
      value /= alive_predators;
    }
  }
  if (alive_prey > 0) {
    for (float &value : prey_dist) {
      value /= alive_prey;
    }
  }

  frame.overlay_stats.step = steps_executed_;
  frame.overlay_stats.max_steps = cfg_.sim_config.max_steps;
  frame.overlay_stats.alive_predators = alive_predators;
  frame.overlay_stats.alive_prey = alive_prey;
  frame.overlay_stats.active_food = active_food;
  frame.overlay_stats.num_species = evolution_.species_count();
  frame.overlay_stats.speed_multiplier =
      cfg_.interactive ? visualization_->speed_multiplier()
                       : cfg_.speed_multiplier;
  frame.overlay_stats.paused = visualization_->is_paused();
  frame.overlay_stats.experiment_name = cfg_.experiment_name;
  frame.overlay_stats.total_kills = event_totals_.kills;
  frame.overlay_stats.total_food_eaten = event_totals_.food_eaten;
  frame.overlay_stats.total_births = event_totals_.births;
  frame.overlay_stats.total_deaths = event_totals_.deaths;
  for (int i = 0; i < 5; ++i) {
    frame.overlay_stats.predator_energy_dist[i] = pred_dist[i];
    frame.overlay_stats.prey_energy_dist[i] = prey_dist[i];
  }

  Entity selected = visualization_->selected_entity();
  if (selected != INVALID_ENTITY && registry_.valid(selected)) {
    size_t idx = registry_.index_of(selected);
    const Genome *genome = evolution_.genome_for(selected);
    if (genome && registry_.vitals().alive[idx]) {
      frame.overlay_stats.selected_agent = static_cast<int>(selected.index);
      frame.overlay_stats.selected_energy = vitals.energy[idx];
      frame.overlay_stats.selected_age = vitals.age[idx];
      frame.overlay_stats.selected_kills = stats.kills[idx];
      frame.overlay_stats.selected_food_eaten = stats.food_eaten[idx];
      frame.overlay_stats.selected_genome_complexity = genome->complexity();
      frame.selected_genome = genome;
      frame.selected_entity = selected;
      frame.has_selected_vision = true;
      frame.selected_position = Vec2{positions.x[idx], positions.y[idx]};
      frame.selected_vision_range = cfg_.sim_config.vision_range;
      frame.selected_node_activations = selected_node_activations_;

      const Vec2 selected_pos{positions.x[idx], positions.y[idx]};
      for (Entity other : living) {
        if (other == selected) {
          continue;
        }

        size_t other_idx = registry_.index_of(other);
        if (!vitals.alive[other_idx]) {
          continue;
        }

        Vec2 other_pos{positions.x[other_idx], positions.y[other_idx]};
        Vec2 diff = other_pos - selected_pos;
        if (diff.length() > cfg_.sim_config.vision_range) {
          continue;
        }

        sf::Color color =
            identity.type[other_idx] == IdentitySoA::TYPE_PREDATOR
                ? sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G,
                            chart_colors::PREDATOR_B, visual::SENSOR_ALPHA)
                : sf::Color(chart_colors::PREY_R, chart_colors::PREY_G,
                            chart_colors::PREY_B, visual::SENSOR_ALPHA);
        frame.sensor_lines.push_back(
            RenderLine{selected_pos, other_pos, color});
      }

      for (const auto &food : frame.foods) {
        Vec2 diff = wrap_diff(food.position - selected_pos,
                              static_cast<float>(cfg_.sim_config.grid_size),
                              static_cast<float>(cfg_.sim_config.grid_size));
        if (diff.length() > cfg_.sim_config.vision_range) {
          continue;
        }

        frame.sensor_lines.push_back(RenderLine{
            selected_pos, food.position,
            sf::Color(chart_colors::FOOD_R, chart_colors::FOOD_G,
                      chart_colors::FOOD_B, visual::FOOD_SENSOR_ALPHA)});
      }
    }
  }

  return frame;
}

void App::accumulate_events(const std::vector<SimEvent> &events) {
  for (const auto &event : events) {
    switch (event.type) {
      case SimEvent::Kill:
        ++event_totals_.kills;
        break;
      case SimEvent::Food:
        ++event_totals_.food_eaten;
        break;
      case SimEvent::Birth:
        ++event_totals_.births;
        break;
      case SimEvent::Death:
        ++event_totals_.deaths;
        break;
    }
  }
}

bool App::should_continue() const {
  if (cfg_.sim_config.max_steps > 0 &&
      steps_executed_ >= cfg_.sim_config.max_steps) {
    return false;
  }

  if (g_running_ == 0) {
    return false;
  }

  return true;
}

void App::log_report(const StepMetrics &snapshot) const {
  spdlog::info(
      "Step {:6d}: predators={} prey={} births={} deaths={} species={}",
      snapshot.step, snapshot.predator_count, snapshot.prey_count,
      snapshot.births, snapshot.deaths, snapshot.num_species);
}

void App::log_early_stop(bool user_quit) const {
  if (user_quit) {
    spdlog::info("Simulation stopped by user (window closed)");
  } else {
    spdlog::info("Simulation stopped by signal (Ctrl+C)");
  }
}

bool App::run() {
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
    while (should_continue()) {
      step();

      if (steps_executed_ % cfg_.sim_config.report_interval_steps == 0) {
        auto snapshot = record_and_log();
        log_report(snapshot);
      }
    }

    if (metrics_.history().empty()) {
      record_and_log();
    }

    logger_.flush();
    completed = (g_running_ != 0);
  } else {
    if (!visualization_) {
      spdlog::error("Visualization requested but not initialized");
      return true;
    }

    while (should_continue()) {
      MOONAI_PROFILE_SCOPE("frame_total");

      {
        MOONAI_PROFILE_SCOPE("handle_events");
        visualization_->handle_events();
      }

      if (visualization_->should_close()) {
        completed = false;
        user_quit = true;
        break;
      }

      if (g_running_ == 0) {
        completed = false;
        break;
      }

      const bool step_requested =
          cfg_.interactive && visualization_->should_step();

      if (cfg_.interactive && visualization_->is_paused() && !step_requested) {
        {
          MOONAI_PROFILE_SCOPE("render");
          visualization_->render(build_frame_snapshot());
        }
        continue;
      }

      if (cfg_.interactive && step_requested) {
        visualization_->clear_step();
      }

      int steps_to_run =
          step_requested
              ? 1
              : (cfg_.interactive ? visualization_->speed_multiplier()
                                  : cfg_.speed_multiplier);
      steps_to_run = std::max(1, steps_to_run);

      for (int i = 0; i < steps_to_run && should_continue(); ++i) {
        step();

        if (steps_executed_ % cfg_.sim_config.report_interval_steps == 0) {
          auto snapshot = record_and_log();
          log_report(snapshot);
        }
      }

      if (cfg_.interactive) {
        MOONAI_PROFILE_SCOPE("update_selected");
        update_selected_visualization();
      }

      {
        MOONAI_PROFILE_SCOPE("render");
        visualization_->render(build_frame_snapshot());
      }
    }

    if (metrics_.history().empty()) {
      record_and_log();
    }

    logger_.flush();
  }

  if (!completed) {
    log_early_stop(user_quit);
  }

  spdlog::info("Output saved to: {}", logger_.run_dir());

  return completed;
}

} // namespace moonai
