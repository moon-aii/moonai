#include "core/config.hpp"

#include <cstdio>
#include <fstream>
#include <spdlog/spdlog.h>
#include <sstream>

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace moonai {

namespace {

std::uint64_t fnv1a_64(const std::string &value) {
  std::uint64_t hash = 1469598103934665603ull;
  for (unsigned char ch : value) {
    hash ^= static_cast<std::uint64_t>(ch);
    hash *= 1099511628211ull;
  }
  return hash;
}

// Read a Lua table field into a C++ variable (silently skips if absent)
template <typename T>
void lua_get(const sol::table &tbl, const char *key, T &field) {
  auto val = tbl[key];
  if (val.valid()) {
    field = val.get<T>();
  }
}

void lua_get_boundary(const sol::table &tbl, const char *key,
                      BoundaryMode &field) {
  auto val = tbl[key];
  if (val.valid()) {
    std::string s = val.get<std::string>();
    if (s == "wrap")
      field = BoundaryMode::Wrap;
    else if (s == "clamp")
      field = BoundaryMode::Clamp;
  }
}

void lua_get_bool(const sol::table &tbl, const char *key, bool &field) {
  auto val = tbl[key];
  if (val.valid()) {
    field = val.get<bool>();
  }
}

void lua_get_uint64(const sol::table &tbl, const char *key,
                    std::uint64_t &field) {
  auto val = tbl[key];
  if (val.valid()) {
    field = static_cast<std::uint64_t>(val.get<double>());
  }
}

// Populate a SimulationConfig from a Lua table
SimulationConfig table_to_config(const sol::table &tbl) {
  SimulationConfig config;

  lua_get(tbl, "grid_size", config.grid_size);
  lua_get_boundary(tbl, "boundary_mode", config.boundary_mode);
  lua_get(tbl, "predator_count", config.predator_count);
  lua_get(tbl, "prey_count", config.prey_count);
  lua_get(tbl, "predator_speed", config.predator_speed);
  lua_get(tbl, "prey_speed", config.prey_speed);
  lua_get(tbl, "vision_range", config.vision_range);
  lua_get(tbl, "attack_range", config.attack_range);
  lua_get(tbl, "initial_energy", config.initial_energy);
  lua_get(tbl, "energy_drain_per_step", config.energy_drain_per_step);
  lua_get(tbl, "energy_gain_from_kill", config.energy_gain_from_kill);
  lua_get(tbl, "energy_gain_from_food", config.energy_gain_from_food);
  lua_get(tbl, "food_pickup_range", config.food_pickup_range);
  lua_get(tbl, "food_count", config.food_count);
  lua_get(tbl, "food_respawn_rate", config.food_respawn_rate);
  lua_get(tbl, "mutation_rate", config.mutation_rate);
  lua_get(tbl, "crossover_rate", config.crossover_rate);
  lua_get(tbl, "weight_mutation_power", config.weight_mutation_power);
  lua_get(tbl, "add_node_rate", config.add_node_rate);
  lua_get(tbl, "add_connection_rate", config.add_connection_rate);
  lua_get(tbl, "delete_connection_rate", config.delete_connection_rate);
  lua_get(tbl, "max_hidden_nodes", config.max_hidden_nodes);
  lua_get(tbl, "max_steps", config.max_steps);
  lua_get(tbl, "compatibility_threshold", config.compatibility_threshold);
  lua_get(tbl, "c1_excess", config.c1_excess);
  lua_get(tbl, "c2_disjoint", config.c2_disjoint);
  lua_get(tbl, "c3_weight", config.c3_weight);
  lua_get(tbl, "species_update_interval_steps",
          config.species_update_interval_steps);
  lua_get(tbl, "target_fps", config.target_fps);
  lua_get_uint64(tbl, "seed", config.seed);
  lua_get(tbl, "output_dir", config.output_dir);
  lua_get(tbl, "report_interval_steps", config.report_interval_steps);
  lua_get(tbl, "mate_range", config.mate_range);
  lua_get(tbl, "reproduction_energy_threshold",
          config.reproduction_energy_threshold);
  lua_get(tbl, "reproduction_energy_cost", config.reproduction_energy_cost);
  lua_get(tbl, "offspring_initial_energy", config.offspring_initial_energy);
  lua_get(tbl, "min_reproductive_age_steps", config.min_reproductive_age_steps);
  lua_get(tbl, "reproduction_cooldown_steps",
          config.reproduction_cooldown_steps);
  lua_get(tbl, "birth_spawn_radius", config.birth_spawn_radius);
  lua_get(tbl, "fitness_survival_weight", config.fitness_survival_weight);
  lua_get(tbl, "fitness_kill_weight", config.fitness_kill_weight);
  lua_get(tbl, "fitness_energy_weight", config.fitness_energy_weight);
  lua_get(tbl, "fitness_distance_weight", config.fitness_distance_weight);
  lua_get(tbl, "complexity_penalty_weight", config.complexity_penalty_weight);
  lua_get(tbl, "activation_function", config.activation_function);

  return config;
}

} // anonymous namespace

// ── Lua config loading ──────────────────────────────────────────────────

std::map<std::string, SimulationConfig>
load_all_configs_lua(const std::string &filepath) {
  std::map<std::string, SimulationConfig> configs;

  sol::state lua;
  lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::table,
                     sol::lib::string);

  // Inject C++ defaults as a Lua global so configs can reference them
  // without depending on a default.lua file.  Lua configs use:
  //   local base = moonai_defaults
  {
    SimulationConfig d;
    sol::table t = lua.create_table();
    t["grid_size"] = d.grid_size;
    t["boundary_mode"] =
        (d.boundary_mode == BoundaryMode::Wrap) ? "wrap" : "clamp";
    t["predator_count"] = d.predator_count;
    t["prey_count"] = d.prey_count;
    t["predator_speed"] = d.predator_speed;
    t["prey_speed"] = d.prey_speed;
    t["vision_range"] = d.vision_range;
    t["attack_range"] = d.attack_range;
    t["initial_energy"] = d.initial_energy;
    t["energy_drain_per_step"] = d.energy_drain_per_step;
    t["energy_gain_from_kill"] = d.energy_gain_from_kill;
    t["energy_gain_from_food"] = d.energy_gain_from_food;
    t["food_pickup_range"] = d.food_pickup_range;
    t["food_count"] = d.food_count;
    t["food_respawn_rate"] = d.food_respawn_rate;
    t["mutation_rate"] = d.mutation_rate;
    t["crossover_rate"] = d.crossover_rate;
    t["weight_mutation_power"] = d.weight_mutation_power;
    t["add_node_rate"] = d.add_node_rate;
    t["add_connection_rate"] = d.add_connection_rate;
    t["delete_connection_rate"] = d.delete_connection_rate;
    t["max_hidden_nodes"] = d.max_hidden_nodes;
    t["max_steps"] = d.max_steps;
    t["compatibility_threshold"] = d.compatibility_threshold;
    t["c1_excess"] = d.c1_excess;
    t["c2_disjoint"] = d.c2_disjoint;
    t["c3_weight"] = d.c3_weight;
    t["species_update_interval_steps"] = d.species_update_interval_steps;
    t["target_fps"] = d.target_fps;
    t["seed"] = static_cast<double>(d.seed);
    t["output_dir"] = d.output_dir;
    t["report_interval_steps"] = d.report_interval_steps;
    t["mate_range"] = d.mate_range;
    t["reproduction_energy_threshold"] = d.reproduction_energy_threshold;
    t["reproduction_energy_cost"] = d.reproduction_energy_cost;
    t["offspring_initial_energy"] = d.offspring_initial_energy;
    t["min_reproductive_age_steps"] = d.min_reproductive_age_steps;
    t["reproduction_cooldown_steps"] = d.reproduction_cooldown_steps;
    t["birth_spawn_radius"] = d.birth_spawn_radius;
    t["fitness_survival_weight"] = d.fitness_survival_weight;
    t["fitness_kill_weight"] = d.fitness_kill_weight;
    t["fitness_energy_weight"] = d.fitness_energy_weight;
    t["fitness_distance_weight"] = d.fitness_distance_weight;
    t["complexity_penalty_weight"] = d.complexity_penalty_weight;
    t["activation_function"] = d.activation_function;
    lua["moonai_defaults"] = t;
  }

  try {
    sol::protected_function_result result = lua.safe_script_file(filepath);
    if (!result.valid()) {
      sol::error err = result;
      spdlog::error("Lua config error in '{}': {}", filepath, err.what());
      return configs;
    }

    sol::object obj = result;
    if (obj.get_type() != sol::type::table) {
      spdlog::error("Lua config '{}' must return a table", filepath);
      return configs;
    }

    sol::table tbl = obj.as<sol::table>();

    for (auto &[key, val] : tbl) {
      if (key.get_type() == sol::type::string &&
          val.get_type() == sol::type::table) {
        std::string name = key.as<std::string>();
        configs[name] = table_to_config(val.as<sol::table>());
      }
    }

    if (configs.empty()) {
      spdlog::error("Lua config '{}' returned no named experiments. "
                    "Expected: return {{ name = {{ ...params... }}, ... }}",
                    filepath);
    } else {
      spdlog::info("Loaded {} experiment(s) from '{}'.", configs.size(),
                   filepath);
    }
    return configs;

  } catch (const std::exception &e) {
    spdlog::error("Failed to load Lua config '{}': {}", filepath, e.what());
    return configs;
  }
}

// ── JSON output (for config snapshots) ──────────────────────────────────

nlohmann::json config_to_json(const SimulationConfig &config) {
  nlohmann::json j;
  j["grid_size"] = config.grid_size;
  j["boundary_mode"] =
      (config.boundary_mode == BoundaryMode::Wrap) ? "wrap" : "clamp";
  j["predator_count"] = config.predator_count;
  j["prey_count"] = config.prey_count;
  j["predator_speed"] = config.predator_speed;
  j["prey_speed"] = config.prey_speed;
  j["vision_range"] = config.vision_range;
  j["attack_range"] = config.attack_range;
  j["initial_energy"] = config.initial_energy;
  j["energy_drain_per_step"] = config.energy_drain_per_step;
  j["energy_gain_from_kill"] = config.energy_gain_from_kill;
  j["energy_gain_from_food"] = config.energy_gain_from_food;
  j["food_pickup_range"] = config.food_pickup_range;
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
  j["target_fps"] = config.target_fps;
  j["seed"] = config.seed;
  j["output_dir"] = config.output_dir;
  j["report_interval_steps"] = config.report_interval_steps;
  j["mate_range"] = config.mate_range;
  j["reproduction_energy_threshold"] = config.reproduction_energy_threshold;
  j["reproduction_energy_cost"] = config.reproduction_energy_cost;
  j["offspring_initial_energy"] = config.offspring_initial_energy;
  j["min_reproductive_age_steps"] = config.min_reproductive_age_steps;
  j["reproduction_cooldown_steps"] = config.reproduction_cooldown_steps;
  j["birth_spawn_radius"] = config.birth_spawn_radius;
  j["fitness_survival_weight"] = config.fitness_survival_weight;
  j["fitness_kill_weight"] = config.fitness_kill_weight;
  j["fitness_energy_weight"] = config.fitness_energy_weight;
  j["fitness_distance_weight"] = config.fitness_distance_weight;
  j["complexity_penalty_weight"] = config.complexity_penalty_weight;
  j["activation_function"] = config.activation_function;

  return j;
}

std::string fingerprint_config(const SimulationConfig &config) {
  const std::string payload = config_to_json(config).dump();
  std::ostringstream oss;
  oss << std::hex << fnv1a_64(payload);
  return oss.str();
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
  check(config.predator_count + config.prey_count <= 50000, "population",
        "total population must be <= 50000");

  // Agent
  check(config.predator_speed > 0.0f, "predator_speed", "must be > 0");
  check(config.prey_speed > 0.0f, "prey_speed", "must be > 0");
  check(config.vision_range > 0.0f, "vision_range", "must be > 0");
  check(config.attack_range > 0.0f, "attack_range", "must be > 0");
  check(config.attack_range < config.vision_range, "attack_range",
        "must be less than vision_range");
  check(config.initial_energy > 0.0f, "initial_energy", "must be > 0");
  check(config.energy_drain_per_step >= 0.0f, "energy_drain_per_step",
        "must be >= 0");

  // Food
  check(config.food_count >= 0, "food_count", "must be >= 0");
  check(config.food_respawn_rate >= 0.0f && config.food_respawn_rate <= 1.0f,
        "food_respawn_rate", "must be in [0, 1]");

  // Evolution rates
  check(config.mutation_rate >= 0.0f && config.mutation_rate <= 1.0f,
        "mutation_rate", "must be in [0, 1]");
  check(config.crossover_rate >= 0.0f && config.crossover_rate <= 1.0f,
        "crossover_rate", "must be in [0, 1]");
  check(config.add_node_rate >= 0.0f && config.add_node_rate <= 1.0f,
        "add_node_rate", "must be in [0, 1]");
  check(config.add_connection_rate >= 0.0f &&
            config.add_connection_rate <= 1.0f,
        "add_connection_rate", "must be in [0, 1]");
  check(config.delete_connection_rate >= 0.0f &&
            config.delete_connection_rate <= 1.0f,
        "delete_connection_rate", "must be in [0, 1]");
  check(config.weight_mutation_power > 0.0f, "weight_mutation_power",
        "must be > 0");
  check(config.max_steps >= 0, "max_steps", "must be >= 0 (0 = infinite)");

  // Speciation
  check(config.compatibility_threshold > 0.0f, "compatibility_threshold",
        "must be > 0");
  check(config.species_update_interval_steps >= 1,
        "species_update_interval_steps", "must be >= 1");

  // Simulation
  check(config.target_fps >= 1 && config.target_fps <= 1000, "target_fps",
        "must be in [1, 1000]");
  check(config.report_interval_steps >= 1, "report_interval_steps",
        "must be >= 1");
  check(config.mate_range > 0.0f, "mate_range", "must be > 0");
  check(config.reproduction_energy_threshold > 0.0f,
        "reproduction_energy_threshold", "must be > 0");
  check(config.reproduction_energy_cost > 0.0f, "reproduction_energy_cost",
        "must be > 0");
  check(config.offspring_initial_energy > 0.0f, "offspring_initial_energy",
        "must be > 0");
  check(config.min_reproductive_age_steps >= 0, "min_reproductive_age_steps",
        "must be >= 0");
  check(config.reproduction_cooldown_steps >= 0, "reproduction_cooldown_steps",
        "must be >= 0");
  check(config.birth_spawn_radius >= 0.0f, "birth_spawn_radius",
        "must be >= 0");

  return errors;
}

// ── CLI override ────────────────────────────────────────────────────────

std::vector<ConfigError> apply_overrides(
    SimulationConfig &config,
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
      else if (key == "target_fps")
        config.target_fps = std::stoi(val);
      else if (key == "report_interval_steps")
        config.report_interval_steps = std::stoi(val);
      else if (key == "min_reproductive_age_steps")
        config.min_reproductive_age_steps = std::stoi(val);
      else if (key == "reproduction_cooldown_steps")
        config.reproduction_cooldown_steps = std::stoi(val);
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
      else if (key == "attack_range")
        config.attack_range = std::stof(val);
      else if (key == "initial_energy")
        config.initial_energy = std::stof(val);
      else if (key == "energy_drain_per_step")
        config.energy_drain_per_step = std::stof(val);
      else if (key == "energy_gain_from_kill")
        config.energy_gain_from_kill = std::stof(val);
      else if (key == "energy_gain_from_food")
        config.energy_gain_from_food = std::stof(val);
      else if (key == "food_pickup_range")
        config.food_pickup_range = std::stof(val);
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
      else if (key == "fitness_survival_weight")
        config.fitness_survival_weight = std::stof(val);
      else if (key == "fitness_kill_weight")
        config.fitness_kill_weight = std::stof(val);
      else if (key == "fitness_energy_weight")
        config.fitness_energy_weight = std::stof(val);
      else if (key == "fitness_distance_weight")
        config.fitness_distance_weight = std::stof(val);
      else if (key == "complexity_penalty_weight")
        config.complexity_penalty_weight = std::stof(val);
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
      else if (key == "boundary_mode") {
        if (val == "wrap")
          config.boundary_mode = BoundaryMode::Wrap;
        else if (val == "clamp")
          config.boundary_mode = BoundaryMode::Clamp;
        else
          errors.push_back({key, "must be 'wrap' or 'clamp'"});
      } else if (key == "output_dir")
        config.output_dir = val;
      else if (key == "activation_function")
        config.activation_function = val;
      // Unknown key
      else {
        errors.push_back({key, "unknown config key"});
      }
    } catch (const std::exception &e) {
      errors.push_back(
          {key, std::string("invalid value '") + val + "': " + e.what()});
    }
  }

  return errors;
}

// ── Float overrides (from Lua hooks) ─────────────────────────────────────

void apply_overrides_float(SimulationConfig &config,
                           const std::map<std::string, float> &overrides) {
  for (const auto &[key, val] : overrides) {
    // Float fields
    if (key == "mutation_rate")
      config.mutation_rate = val;
    else if (key == "crossover_rate")
      config.crossover_rate = val;
    else if (key == "weight_mutation_power")
      config.weight_mutation_power = val;
    else if (key == "add_node_rate")
      config.add_node_rate = val;
    else if (key == "add_connection_rate")
      config.add_connection_rate = val;
    else if (key == "delete_connection_rate")
      config.delete_connection_rate = val;
    else if (key == "compatibility_threshold")
      config.compatibility_threshold = val;
    else if (key == "c1_excess")
      config.c1_excess = val;
    else if (key == "c2_disjoint")
      config.c2_disjoint = val;
    else if (key == "c3_weight")
      config.c3_weight = val;
    else if (key == "predator_speed")
      config.predator_speed = val;
    else if (key == "prey_speed")
      config.prey_speed = val;
    else if (key == "vision_range")
      config.vision_range = val;
    else if (key == "attack_range")
      config.attack_range = val;
    else if (key == "initial_energy")
      config.initial_energy = val;
    else if (key == "energy_drain_per_step")
      config.energy_drain_per_step = val;
    else if (key == "energy_gain_from_kill")
      config.energy_gain_from_kill = val;
    else if (key == "energy_gain_from_food")
      config.energy_gain_from_food = val;
    else if (key == "food_respawn_rate")
      config.food_respawn_rate = val;
    else if (key == "fitness_survival_weight")
      config.fitness_survival_weight = val;
    else if (key == "fitness_kill_weight")
      config.fitness_kill_weight = val;
    else if (key == "fitness_energy_weight")
      config.fitness_energy_weight = val;
    else if (key == "fitness_distance_weight")
      config.fitness_distance_weight = val;
    else if (key == "complexity_penalty_weight")
      config.complexity_penalty_weight = val;
    else if (key == "mate_range")
      config.mate_range = val;
    else if (key == "reproduction_energy_threshold")
      config.reproduction_energy_threshold = val;
    else if (key == "reproduction_energy_cost")
      config.reproduction_energy_cost = val;
    else if (key == "offspring_initial_energy")
      config.offspring_initial_energy = val;
    else if (key == "birth_spawn_radius")
      config.birth_spawn_radius = val;
    // Integer fields (truncated from float)
    else if (key == "max_steps")
      config.max_steps = static_cast<int>(val);
    else if (key == "species_update_interval_steps")
      config.species_update_interval_steps = static_cast<int>(val);
    else if (key == "max_hidden_nodes")
      config.max_hidden_nodes = static_cast<int>(val);
    else if (key == "report_interval_steps")
      config.report_interval_steps = static_cast<int>(val);
    else if (key == "min_reproductive_age_steps")
      config.min_reproductive_age_steps = static_cast<int>(val);
    else if (key == "reproduction_cooldown_steps")
      config.reproduction_cooldown_steps = static_cast<int>(val);
    else {
      spdlog::warn("Lua hook returned unknown override key: {}", key);
    }
  }
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
    } else if (arg == "--resume" && i + 1 < argc) {
      args.resume_path = argv[++i];
    } else if (arg == "--checkpoint" && i + 1 < argc) {
      try {
        args.checkpoint_interval = std::stoi(argv[++i]);
      } catch (const std::exception &) {
        std::fprintf(stderr, "Invalid checkpoint interval '%s'\n", argv[i]);
        args.help = true;
      }
    } else if (arg == "--no-gpu") {
      args.no_gpu = true;
    } else if (arg == "--compare" && i + 2 < argc) {
      args.compare_a = argv[++i];
      args.compare_b = argv[++i];
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
        std::fprintf(stderr, "Invalid --set format '%s' (expected key=value)\n",
                     kv.c_str());
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
  fmt::print(
      "MoonAI - Predator-Prey Evolutionary Simulation\n"
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
