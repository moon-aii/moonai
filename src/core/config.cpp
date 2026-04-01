#include "core/config.hpp"

#include <cstdio>
#include <fstream>
#include <spdlog/spdlog.h>

namespace moonai {

// ── JSON output (for config snapshots) ──────────────────────────────────

nlohmann::json config_to_json(const SimulationConfig &config) {
  nlohmann::json j;
  j["grid_size"] = config.grid_size;
  j["predator_count"] = config.predator_count;
  j["prey_count"] = config.prey_count;
  j["predator_speed"] = config.predator_speed;
  j["prey_speed"] = config.prey_speed;
  j["vision_range"] = config.vision_range;
  j["interaction_range"] = config.interaction_range;
  j["initial_energy"] = config.initial_energy;
  j["energy_drain_per_step"] = config.energy_drain_per_step;
  j["energy_gain_from_kill"] = config.energy_gain_from_kill;
  j["energy_gain_from_food"] = config.energy_gain_from_food;
  j["food_count"] = config.food_count;
  j["food_respawn_rate"] = config.food_respawn_rate;
  j["mutation_rate"] = config.mutation_rate;
  j["crossover_rate"] = config.crossover_rate;
  j["weight_mutation_power"] = config.weight_mutation_power;
  j["add_node_rate"] = config.add_node_rate;
  j["add_connection_rate"] = config.add_connection_rate;
  j["delete_connection_rate"] = config.delete_connection_rate;
  j["max_hidden_nodes"] = config.max_hidden_nodes;
  j["max_steps"] = config.max_steps;
  j["compatibility_threshold"] = config.compatibility_threshold;
  j["c1_excess"] = config.c1_excess;
  j["c2_disjoint"] = config.c2_disjoint;
  j["c3_weight"] = config.c3_weight;
  j["species_update_interval_steps"] = config.species_update_interval_steps;
  j["seed"] = config.seed;
  j["output_dir"] = config.output_dir;
  j["report_interval_steps"] = config.report_interval_steps;
  j["mate_range"] = config.mate_range;
  j["reproduction_energy_threshold"] = config.reproduction_energy_threshold;
  j["reproduction_energy_cost"] = config.reproduction_energy_cost;
  j["offspring_initial_energy"] = config.offspring_initial_energy;
  j["birth_spawn_radius"] = config.birth_spawn_radius;
  return j;
}

void save_config(const SimulationConfig &config, const std::string &filepath) {
  const nlohmann::json j = config_to_json(config);

  std::ofstream file(filepath);
  file << j.dump(4);
  spdlog::info("Config saved to '{}'.", filepath);
}

// ── Validation ──────────────────────────────────────────────────────────

std::vector<ConfigError> validate_config(const SimulationConfig &config) {
  std::vector<ConfigError> errors;

  auto check = [&](bool condition, const char *field, const char *msg) {
    if (!condition) {
      errors.push_back({field, msg});
    }
  };

  // Environment
  check(config.grid_size >= 100, "grid_size", "must be >= 100");
  check(config.grid_size <= 20000, "grid_size", "must be <= 20000");

  // Population
  check(config.predator_count >= 1, "predator_count", "must be >= 1");
  check(config.prey_count >= 1, "prey_count", "must be >= 1");
  check(config.predator_count + config.prey_count <= 50000, "population", "total population must be <= 50000");

  // Agent
  check(config.predator_speed > 0.0f, "predator_speed", "must be > 0");
  check(config.prey_speed > 0.0f, "prey_speed", "must be > 0");
  check(config.vision_range > 0.0f, "vision_range", "must be > 0");
  check(config.interaction_range > 0.0f, "interaction_range", "must be > 0");
  check(config.interaction_range < config.vision_range, "interaction_range", "must be less than vision_range");
  check(config.initial_energy > 0.0f, "initial_energy", "must be > 0");
  check(config.energy_drain_per_step >= 0.0f, "energy_drain_per_step", "must be >= 0");

  // Food
  check(config.food_count >= 0, "food_count", "must be >= 0");
  check(config.food_respawn_rate >= 0.0f && config.food_respawn_rate <= 1.0f, "food_respawn_rate", "must be in [0, 1]");

  // Evolution rates
  check(config.mutation_rate >= 0.0f && config.mutation_rate <= 1.0f, "mutation_rate", "must be in [0, 1]");
  check(config.crossover_rate >= 0.0f && config.crossover_rate <= 1.0f, "crossover_rate", "must be in [0, 1]");
  check(config.add_node_rate >= 0.0f && config.add_node_rate <= 1.0f, "add_node_rate", "must be in [0, 1]");
  check(config.add_connection_rate >= 0.0f && config.add_connection_rate <= 1.0f, "add_connection_rate",
        "must be in [0, 1]");
  check(config.delete_connection_rate >= 0.0f && config.delete_connection_rate <= 1.0f, "delete_connection_rate",
        "must be in [0, 1]");
  check(config.weight_mutation_power > 0.0f, "weight_mutation_power", "must be > 0");
  check(config.max_steps >= 0, "max_steps", "must be >= 0 (0 = infinite)");

  // Speciation
  check(config.compatibility_threshold > 0.0f, "compatibility_threshold", "must be > 0");
  check(config.species_update_interval_steps >= 1, "species_update_interval_steps", "must be >= 1");

  // Simulation
  check(config.report_interval_steps >= 1, "report_interval_steps", "must be >= 1");
  check(config.mate_range > 0.0f, "mate_range", "must be > 0");
  check(config.reproduction_energy_threshold > 0.0f, "reproduction_energy_threshold", "must be > 0");
  check(config.reproduction_energy_cost > 0.0f, "reproduction_energy_cost", "must be > 0");
  check(config.offspring_initial_energy > 0.0f, "offspring_initial_energy", "must be > 0");
  check(config.birth_spawn_radius >= 0.0f, "birth_spawn_radius", "must be >= 0");

  return errors;
}

// ── CLI override ────────────────────────────────────────────────────────

std::vector<ConfigError> apply_overrides(SimulationConfig &config,
                                         const std::vector<std::pair<std::string, std::string>> &overrides) {
  std::vector<ConfigError> errors;

  for (const auto &[key, val] : overrides) {
    try {
      // Integer fields
      if (key == "grid_size")
        config.grid_size = std::stoi(val);
      else if (key == "predator_count")
        config.predator_count = std::stoi(val);
      else if (key == "prey_count")
        config.prey_count = std::stoi(val);
      else if (key == "food_count")
        config.food_count = std::stoi(val);
      else if (key == "max_hidden_nodes")
        config.max_hidden_nodes = std::stoi(val);
      else if (key == "max_steps")
        config.max_steps = std::stoi(val);
      else if (key == "species_update_interval_steps")
        config.species_update_interval_steps = std::stoi(val);
      else if (key == "report_interval_steps")
        config.report_interval_steps = std::stoi(val);
      // uint64 fields
      else if (key == "seed")
        config.seed = std::stoull(val);
      // Float fields
      else if (key == "predator_speed")
        config.predator_speed = std::stof(val);
      else if (key == "prey_speed")
        config.prey_speed = std::stof(val);
      else if (key == "vision_range")
        config.vision_range = std::stof(val);
      else if (key == "interaction_range")
        config.interaction_range = std::stof(val);
      else if (key == "initial_energy")
        config.initial_energy = std::stof(val);
      else if (key == "energy_drain_per_step")
        config.energy_drain_per_step = std::stof(val);
      else if (key == "energy_gain_from_kill")
        config.energy_gain_from_kill = std::stof(val);
      else if (key == "energy_gain_from_food")
        config.energy_gain_from_food = std::stof(val);
      else if (key == "food_respawn_rate")
        config.food_respawn_rate = std::stof(val);
      else if (key == "mutation_rate")
        config.mutation_rate = std::stof(val);
      else if (key == "crossover_rate")
        config.crossover_rate = std::stof(val);
      else if (key == "weight_mutation_power")
        config.weight_mutation_power = std::stof(val);
      else if (key == "add_node_rate")
        config.add_node_rate = std::stof(val);
      else if (key == "add_connection_rate")
        config.add_connection_rate = std::stof(val);
      else if (key == "delete_connection_rate")
        config.delete_connection_rate = std::stof(val);
      else if (key == "compatibility_threshold")
        config.compatibility_threshold = std::stof(val);
      else if (key == "c1_excess")
        config.c1_excess = std::stof(val);
      else if (key == "c2_disjoint")
        config.c2_disjoint = std::stof(val);
      else if (key == "c3_weight")
        config.c3_weight = std::stof(val);
      else if (key == "mate_range")
        config.mate_range = std::stof(val);
      else if (key == "reproduction_energy_threshold")
        config.reproduction_energy_threshold = std::stof(val);
      else if (key == "reproduction_energy_cost")
        config.reproduction_energy_cost = std::stof(val);
      else if (key == "offspring_initial_energy")
        config.offspring_initial_energy = std::stof(val);
      else if (key == "birth_spawn_radius")
        config.birth_spawn_radius = std::stof(val);
      // String fields
      else if (key == "output_dir")
        config.output_dir = val;
      // Unknown key
      else {
        errors.push_back({key, "unknown config key"});
      }
    } catch (const std::exception &e) {
      errors.push_back({key, std::string("invalid value '") + val + "': " + e.what()});
    }
  }

  return errors;
}
// ── CLI parsing ─────────────────────────────────────────────────────────

CLIArgs parse_args(int argc, const char *argv[]) {
  CLIArgs args;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      args.help = true;
    } else if (arg == "--headless") {
      args.headless = true;
    } else if (arg == "-v" || arg == "--verbose") {
      args.verbose = true;
    } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
      try {
        args.seed_override = std::stoull(argv[++i]);
      } catch (const std::exception &) {
        std::fprintf(stderr, "Invalid seed value '%s'\n", argv[i]);
        args.help = true;
      }
    } else if ((arg == "-n" || arg == "--steps") && i + 1 < argc) {
      try {
        args.max_steps_override = std::stoi(argv[++i]);
      } catch (const std::exception &) {
        std::fprintf(stderr, "Invalid steps value '%s'\n", argv[i]);
        args.help = true;
      }
    } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
      args.config_path = argv[++i];
    } else if (arg == "--no-gpu") {
      args.no_gpu = true;
    }
    // New flags
    else if (arg == "--experiment" && i + 1 < argc) {
      args.experiment_name = argv[++i];
    } else if (arg == "--all") {
      args.run_all = true;
    } else if (arg == "--list") {
      args.list_experiments = true;
    } else if (arg == "--name" && i + 1 < argc) {
      args.run_name = argv[++i];
    } else if (arg == "--validate") {
      args.validate_only = true;
    } else if (arg == "--set" && i + 1 < argc) {
      std::string kv = argv[++i];
      auto eq = kv.find('=');
      if (eq == std::string::npos) {
        std::fprintf(stderr, "Invalid --set format '%s' (expected key=value)\n", kv.c_str());
        args.help = true;
      } else {
        args.overrides.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
      }
    }
    // Positional argument: config path
    else if (arg[0] != '-') {
      args.config_path = arg;
    } else {
      spdlog::warn("Unknown argument: {}", arg);
    }
  }

  return args;
}

void print_usage(const char *program_name) {
  fmt::print("MoonAI - Predator-Prey Evolutionary Simulation\n"
             "\n"
             "Usage: {} [OPTIONS] [config.lua]\n"
             "\n"
             "Options:\n"
             "  -c, --config <path>       Path to Lua config (default: "
             "config/default.lua)\n"
             "  -s, --seed <number>       Override random seed\n"
             "  -n, --steps <n>           Override max steps (0 = infinite)\n"
             "      --headless            Run without visualization\n"
             "  -v, --verbose             Enable debug logging\n"
             "      --resume <path>       Resume from a checkpoint JSON file\n"
             "      --checkpoint <n>      Save checkpoint every N steps (0 = "
             "disabled)\n"
             "      --compare <a> <b>     Print structural diff between two genome "
             "JSON files\n"
             "      --no-gpu              Disable CUDA GPU acceleration (use CPU path)"
#ifdef MOONAI_ENABLE_CUDA
             " (cuda compiled in)\n"
#else
             " (no-cuda build: always CPU)\n"
#endif
             "\n"
             "Experiment orchestration:\n"
             "      --experiment <name>   Select one experiment from a multi-config "
             "Lua file\n"
             "      --all                 Run all experiments sequentially (headless "
             "only)\n"
             "      --list                List experiment names and exit\n"
             "      --name <name>         Override output directory name\n"
             "      --set key=value       Override a config parameter (repeatable)\n"
             "      --validate            Validate config and exit\n"
             "\n"
             "  -h, --help                Show this help message\n",
             program_name);
}

} // namespace moonai
