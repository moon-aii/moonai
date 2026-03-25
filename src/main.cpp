#include "core/config.hpp"
#include "core/random.hpp"
#include "data/logger.hpp"
#include "data/metrics.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/components.hpp"
#include "simulation/registry.hpp"
#include "simulation/simulation_manager.hpp"
#include "visualization/visualization_manager.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>

namespace {

volatile std::sig_atomic_t g_running = 1;

void signal_handler(int) {
  g_running = 0;
}

int run_experiment(const std::string &name, moonai::SimulationConfig config,
                   const moonai::CLIArgs &args) {
  if (args.seed_override != 0) {
    config.seed = args.seed_override;
  }
  if (args.max_steps_override != 0) {
    config.max_steps = args.max_steps_override;
  }

  const auto errors = moonai::validate_config(config);
  if (!errors.empty()) {
    for (const auto &error : errors) {
      spdlog::error("Config error [{}]: {}", error.field, error.message);
    }
    return 1;
  }

  if (config.seed == 0) {
    config.seed = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
  }

  moonai::Random rng(config.seed);
  moonai::Registry registry;
  moonai::SimulationManager simulation(config);
  moonai::EvolutionManager evolution(config, rng);
  moonai::Logger logger(config.output_dir, config.seed,
                        args.run_name.empty() ? name : args.run_name);
  moonai::MetricsCollector metrics;

  simulation.initialize();
  evolution.initialize(moonai::SensorSoA::INPUT_COUNT, 2);
  evolution.seed_initial_population_ecs(registry);
  simulation.refresh_state_ecs(registry);
  logger.initialize(config);

  bool headless = args.headless;
  if (!headless && std::getenv("DISPLAY") == nullptr &&
      std::getenv("WAYLAND_DISPLAY") == nullptr) {
    headless = true;
    spdlog::warn("No display server found; switching to headless mode.");
  }

  moonai::VisualizationManager visualization(config);
  if (!headless) {
    visualization.initialize();
  }

  const float dt = 1.0f / static_cast<float>(config.target_fps);
  int steps_executed = 0;
  int births_in_window = 0;
  int deaths_in_window = 0;
  std::vector<moonai::Vec2> actions;

  auto record_window = [&]() {
    evolution.refresh_species_ecs(registry);

    float best_predator = 0.0f, avg_predator = 0.0f;
    float best_prey = 0.0f, avg_prey = 0.0f;
    evolution.get_fitness_by_type_ecs(registry, best_predator, avg_predator,
                                      best_prey, avg_prey);

    int alive_predators = 0, alive_prey = 0;
    for (moonai::Entity e : registry.living_entities()) {
      size_t idx = registry.index_of(e);
      if (registry.vitals().alive[idx]) {
        if (registry.identity().type[idx] ==
            moonai::IdentitySoA::TYPE_PREDATOR) {
          alive_predators++;
        } else {
          alive_prey++;
        }
      }
    }

    auto snapshot = metrics.collect_ecs(steps_executed, registry, evolution,
                                        births_in_window, deaths_in_window,
                                        evolution.species_count());
    logger.log_report(snapshot);

    // Find best genome
    const moonai::Genome *best_genome = nullptr;
    float best_fitness = -std::numeric_limits<float>::infinity();
    for (moonai::Entity e : registry.living_entities()) {
      const auto *genome = evolution.genome_for(e);
      if (genome && genome->fitness() > best_fitness) {
        best_fitness = genome->fitness();
        best_genome = genome;
      }
    }
    if (best_genome) {
      logger.log_best_genome(steps_executed, *best_genome);
    }

    logger.log_species(steps_executed, evolution.species());
    logger.flush();
    births_in_window = 0;
    deaths_in_window = 0;
    return snapshot;
  };

  auto update_selected_visualization = [&]() {
    moonai::Entity selected = visualization.selected_entity();
    if (selected == moonai::INVALID_ENTITY || !registry.valid(selected)) {
      return;
    }

    size_t idx = registry.index_of(selected);
    const auto *genome = evolution.genome_for(selected);
    if (!genome || !registry.vitals().alive[idx]) {
      return;
    }

    // Activate network with current sensors
    const float *sensors = registry.sensors().input_ptr(idx);
    std::vector<float> sensor_vec(sensors,
                                  sensors + moonai::SensorSoA::INPUT_COUNT);

    moonai::NeuralNetwork *network =
        evolution.network_cache().get_network(selected);
    if (network) {
      network->activate(sensor_vec);
      visualization.set_selected_activations(network->last_activations(),
                                             network->node_index_map());
    }
  };

  auto advance_one_step = [&]() {
    // Get actions from neural networks
    evolution.compute_actions_ecs(registry, actions);

    // Apply actions to movement
    size_t action_idx = 0;
    for (moonai::Entity e : registry.living_entities()) {
      size_t idx = registry.index_of(e);
      if (!registry.vitals().alive[idx]) {
        continue;
      }

      if (action_idx < actions.size()) {
        // Apply action to velocity
        float dx = actions[action_idx].x;
        float dy = actions[action_idx].y;
        float speed = registry.motion().speed[idx];

        registry.motion().vel_x[idx] = dx * speed;
        registry.motion().vel_y[idx] = dy * speed;

        // Update position
        registry.positions().x[idx] += registry.motion().vel_x[idx] * dt;
        registry.positions().y[idx] += registry.motion().vel_y[idx] * dt;

        // Track distance traveled
        registry.stats().distance_traveled[idx] +=
            std::sqrt(dx * dx + dy * dy) * speed * dt;

        action_idx++;
      }
    }

    // Run simulation step
    simulation.step_ecs(registry, dt);

    // Process reproduction
    auto pairs = simulation.find_reproduction_pairs_ecs(registry);
    for (const auto &pair : pairs) {
      moonai::Entity child = evolution.create_offspring_ecs(
          registry, pair.parent_a, pair.parent_b, pair.spawn_position);
      if (child != moonai::INVALID_ENTITY) {
        births_in_window++;
      }
    }

    // Update fitness
    evolution.refresh_fitness_ecs(registry);

    // Update species periodically
    if (config.species_update_interval_steps > 0 &&
        (steps_executed % config.species_update_interval_steps) == 0) {
      evolution.refresh_species_ecs(registry);
    }

    // Count events
    for (const auto &event : simulation.last_events()) {
      if (event.type == moonai::SimEvent::Death) {
        deaths_in_window++;
      }
    }

    ++steps_executed;

    // Log per-step data if enabled
    if (config.step_log_enabled &&
        (steps_executed % config.step_log_interval) == 0) {
      logger.log_step(steps_executed, registry);
    }
    logger.log_events(steps_executed, simulation.last_events());

    // Log report window
    if ((steps_executed % config.report_interval_steps) == 0) {
      const auto snapshot = record_window();
      spdlog::info(
          "Step {:>6d}: predators={} prey={} births={} deaths={} species={}",
          steps_executed, simulation.alive_predators(), simulation.alive_prey(),
          snapshot.births, snapshot.deaths, evolution.species_count());
    }
  };

  while (g_running &&
         (config.max_steps == 0 || steps_executed < config.max_steps)) {
    if (headless) {
      advance_one_step();
      continue;
    }

    visualization.handle_events();
    if (visualization.should_close()) {
      break;
    }

    if (visualization.is_paused() && !visualization.should_step()) {
      visualization.render_ecs(registry, evolution, simulation, steps_executed);
      continue;
    }
    visualization.clear_step();

    const int frame_steps = std::max(1, visualization.speed_multiplier());
    for (int i = 0;
         i < frame_steps &&
         (config.max_steps == 0 || steps_executed < config.max_steps) &&
         g_running;
         ++i) {
      advance_one_step();
    }

    update_selected_visualization();

    visualization.render_ecs(registry, evolution, simulation, steps_executed);
  }

  if (births_in_window > 0 || deaths_in_window > 0 ||
      metrics.history().empty()) {
    record_window();
  }

  logger.flush();
  spdlog::info("Output saved to: {}", logger.run_dir());
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  const auto args = moonai::parse_args(argc, argv);
  if (args.help) {
    moonai::print_usage(argv[0]);
    return 0;
  }

  spdlog::set_level(args.verbose ? spdlog::level::debug : spdlog::level::info);
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  auto configs = moonai::load_all_configs_lua(args.config_path);
  if (configs.empty()) {
    spdlog::error("No configs loaded from '{}'", args.config_path);
    return 1;
  }

  if (args.list_experiments) {
    for (const auto &[config_name, _] : configs) {
      std::printf("%s\n", config_name.c_str());
    }
    return 0;
  }

  if (args.validate_only) {
    moonai::SimulationConfig config;
    if (!args.experiment_name.empty()) {
      config = configs.at(args.experiment_name);
    } else {
      config = configs.begin()->second;
    }
    const auto errors = moonai::validate_config(config);
    if (errors.empty()) {
      std::printf("OK\n");
      return 0;
    }
    for (const auto &error : errors) {
      std::fprintf(stderr, "ERROR [%s]: %s\n", error.field.c_str(),
                   error.message.c_str());
    }
    return 1;
  }

  if (args.run_all) {
    int failures = 0;
    for (const auto &[config_name, config] : configs) {
      failures += run_experiment(config_name, config, args);
    }
    return failures == 0 ? 0 : 1;
  }

  std::string selected = args.experiment_name;
  if (selected.empty()) {
    selected = configs.begin()->first;
    if (configs.size() > 1) {
      spdlog::warn("Multiple experiments found; using '{}'.", selected);
    }
  }

  auto it = configs.find(selected);
  if (it == configs.end()) {
    spdlog::error("Experiment '{}' not found.", selected);
    return 1;
  }

  return run_experiment(selected, it->second, args);
}
