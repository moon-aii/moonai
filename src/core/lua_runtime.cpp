#include "core/lua_runtime.hpp"

#include <spdlog/spdlog.h>

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace moonai {

// ── Lua table helpers (duplicated from config.cpp — anonymous namespace) ──

namespace {

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

void inject_defaults(sol::state &lua) {
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

sol::table config_to_table(sol::state &lua, const SimulationConfig &c) {
  sol::table t = lua.create_table();
  t["grid_size"] = c.grid_size;
  t["boundary_mode"] =
      (c.boundary_mode == BoundaryMode::Wrap) ? "wrap" : "clamp";
  t["predator_count"] = c.predator_count;
  t["prey_count"] = c.prey_count;
  t["predator_speed"] = c.predator_speed;
  t["prey_speed"] = c.prey_speed;
  t["vision_range"] = c.vision_range;
  t["attack_range"] = c.attack_range;
  t["initial_energy"] = c.initial_energy;
  t["energy_drain_per_step"] = c.energy_drain_per_step;
  t["energy_gain_from_kill"] = c.energy_gain_from_kill;
  t["energy_gain_from_food"] = c.energy_gain_from_food;
  t["food_pickup_range"] = c.food_pickup_range;
  t["food_count"] = c.food_count;
  t["food_respawn_rate"] = c.food_respawn_rate;
  t["mutation_rate"] = c.mutation_rate;
  t["crossover_rate"] = c.crossover_rate;
  t["weight_mutation_power"] = c.weight_mutation_power;
  t["add_node_rate"] = c.add_node_rate;
  t["add_connection_rate"] = c.add_connection_rate;
  t["delete_connection_rate"] = c.delete_connection_rate;
  t["max_hidden_nodes"] = c.max_hidden_nodes;
  t["max_steps"] = c.max_steps;
  t["compatibility_threshold"] = c.compatibility_threshold;
  t["c1_excess"] = c.c1_excess;
  t["c2_disjoint"] = c.c2_disjoint;
  t["c3_weight"] = c.c3_weight;
  t["species_update_interval_steps"] = c.species_update_interval_steps;
  t["target_fps"] = c.target_fps;
  t["seed"] = static_cast<double>(c.seed);
  t["output_dir"] = c.output_dir;
  t["report_interval_steps"] = c.report_interval_steps;
  t["mate_range"] = c.mate_range;
  t["reproduction_energy_threshold"] = c.reproduction_energy_threshold;
  t["reproduction_energy_cost"] = c.reproduction_energy_cost;
  t["offspring_initial_energy"] = c.offspring_initial_energy;
  t["min_reproductive_age_steps"] = c.min_reproductive_age_steps;
  t["reproduction_cooldown_steps"] = c.reproduction_cooldown_steps;
  t["birth_spawn_radius"] = c.birth_spawn_radius;
  t["fitness_survival_weight"] = c.fitness_survival_weight;
  t["fitness_kill_weight"] = c.fitness_kill_weight;
  t["fitness_energy_weight"] = c.fitness_energy_weight;
  t["fitness_distance_weight"] = c.fitness_distance_weight;
  t["complexity_penalty_weight"] = c.complexity_penalty_weight;
  t["activation_function"] = c.activation_function;
  return t;
}

} // anonymous namespace

// ── PIMPL ────────────────────────────────────────────────────────────────────

struct LuaRuntime::Impl {
  sol::state lua;

  struct ExperimentFunctions {
    LuaCallbacks flags;
    sol::protected_function fitness_fn;
    sol::protected_function on_report_window_end;
    sol::protected_function on_experiment_start;
    sol::protected_function on_experiment_end;
  };

  std::map<std::string, ExperimentFunctions> experiment_fns;
  std::string selected;
  LuaCallbacks empty_callbacks; // returned when no experiment selected
};

// ── Lifecycle ────────────────────────────────────────────────────────────────

LuaRuntime::LuaRuntime() : impl_(std::make_unique<Impl>()) {}

LuaRuntime::~LuaRuntime() = default;

// ── Config loading ───────────────────────────────────────────────────────────

std::map<std::string, SimulationConfig>
LuaRuntime::load_config(const std::string &filepath) {
  std::map<std::string, SimulationConfig> configs;

  auto &lua = impl_->lua;
  lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::table,
                     sol::lib::string);
  inject_defaults(lua);

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
      if (key.get_type() != sol::type::string ||
          val.get_type() != sol::type::table)
        continue;

      std::string name = key.as<std::string>();
      sol::table exp_tbl = val.as<sol::table>();

      configs[name] = table_to_config(exp_tbl);

      // Extract Lua callback functions from the experiment table
      Impl::ExperimentFunctions ef;

      auto fit = exp_tbl["fitness_fn"];
      if (fit.valid() && fit.get_type() == sol::type::function) {
        ef.fitness_fn = fit.get<sol::protected_function>();
        ef.flags.has_fitness_fn = true;
      }

      auto window_end = exp_tbl["on_report_window_end"];
      if (window_end.valid() && window_end.get_type() == sol::type::function) {
        ef.on_report_window_end = window_end.get<sol::protected_function>();
        ef.flags.has_on_report_window_end = true;
      }

      auto exp_start = exp_tbl["on_experiment_start"];
      if (exp_start.valid() && exp_start.get_type() == sol::type::function) {
        ef.on_experiment_start = exp_start.get<sol::protected_function>();
        ef.flags.has_on_experiment_start = true;
      }

      auto exp_end = exp_tbl["on_experiment_end"];
      if (exp_end.valid() && exp_end.get_type() == sol::type::function) {
        ef.on_experiment_end = exp_end.get<sol::protected_function>();
        ef.flags.has_on_experiment_end = true;
      }

      impl_->experiment_fns[name] = std::move(ef);
    }

    if (configs.empty()) {
      spdlog::error("Lua config '{}' returned no named experiments.", filepath);
    } else {
      spdlog::info("Loaded {} experiment(s) from '{}'.", configs.size(),
                   filepath);
    }

  } catch (const std::exception &e) {
    spdlog::error("Failed to load Lua config '{}': {}", filepath, e.what());
  }

  return configs;
}

void LuaRuntime::select_experiment(const std::string &name) {
  impl_->selected = name;
  if (impl_->experiment_fns.count(name)) {
    const auto &flags = impl_->experiment_fns[name].flags;
    if (flags.has_fitness_fn)
      spdlog::info("Lua fitness_fn active for '{}'", name);
    if (flags.has_on_report_window_end)
      spdlog::info("Lua on_report_window_end hook active for '{}'", name);
    if (flags.has_on_experiment_start)
      spdlog::info("Lua on_experiment_start hook active for '{}'", name);
    if (flags.has_on_experiment_end)
      spdlog::info("Lua on_experiment_end hook active for '{}'", name);
  }
}

const LuaCallbacks &LuaRuntime::callbacks() const {
  auto it = impl_->experiment_fns.find(impl_->selected);
  if (it != impl_->experiment_fns.end()) {
    return it->second.flags;
  }
  return impl_->empty_callbacks;
}

// ── Lua fitness function ─────────────────────────────────────────────────────

float LuaRuntime::call_fitness(float age_ratio, float kills_or_food,
                               float energy_ratio, float alive_bonus,
                               float dist_ratio, float complexity,
                               const SimulationConfig &config) {
  auto it = impl_->experiment_fns.find(impl_->selected);
  if (it == impl_->experiment_fns.end() || !it->second.flags.has_fitness_fn)
    return -1.0f; // sentinel: caller should use default formula

  auto &lua = impl_->lua;
  sol::table stats = lua.create_table();
  stats["age_ratio"] = age_ratio;
  stats["kills_or_food"] = kills_or_food;
  stats["energy_ratio"] = energy_ratio;
  stats["alive_bonus"] = alive_bonus;
  stats["dist_ratio"] = dist_ratio;
  stats["complexity"] = complexity;

  sol::table weights = lua.create_table();
  weights["survival"] = config.fitness_survival_weight;
  weights["kill"] = config.fitness_kill_weight;
  weights["energy"] = config.fitness_energy_weight;
  weights["distance"] = config.fitness_distance_weight;
  weights["complexity_penalty"] = config.complexity_penalty_weight;

  auto result = it->second.fitness_fn(stats, weights);
  if (!result.valid()) {
    sol::error err = result;
    spdlog::error("Lua fitness_fn error: {}", err.what());
    return 0.0f;
  }
  return std::max(0.0f, result.get<float>());
}

// ── Event hooks ──────────────────────────────────────────────────────────────

bool LuaRuntime::call_on_report_window_end(
    const ReportWindowStats &stats_in,
    std::map<std::string, float> &overrides) {
  auto it = impl_->experiment_fns.find(impl_->selected);
  if (it == impl_->experiment_fns.end() ||
      !it->second.flags.has_on_report_window_end)
    return false;

  auto &lua = impl_->lua;
  sol::table stats = lua.create_table();
  stats["step"] = stats_in.step;
  stats["window_index"] = stats_in.window_index;
  stats["best_fitness"] = stats_in.best_fitness;
  stats["avg_fitness"] = stats_in.avg_fitness;
  stats["num_species"] = stats_in.num_species;
  stats["alive_predators"] = stats_in.alive_predators;
  stats["alive_prey"] = stats_in.alive_prey;
  stats["avg_complexity"] = stats_in.avg_complexity;

  auto result = it->second.on_report_window_end(stats_in.step,
                                                stats_in.window_index, stats);
  if (!result.valid()) {
    sol::error err = result;
    spdlog::error("Lua on_report_window_end error: {}", err.what());
    return false;
  }

  sol::object ret = result;
  if (ret.get_type() != sol::type::table)
    return false;

  sol::table tbl = ret.as<sol::table>();
  for (auto &[key, val] : tbl) {
    if (key.get_type() == sol::type::string &&
        val.get_type() == sol::type::number) {
      overrides[key.as<std::string>()] = val.as<float>();
    }
  }
  return !overrides.empty();
}

void LuaRuntime::call_on_experiment_start(const SimulationConfig &config) {
  auto it = impl_->experiment_fns.find(impl_->selected);
  if (it == impl_->experiment_fns.end() ||
      !it->second.flags.has_on_experiment_start)
    return;

  sol::table cfg = config_to_table(impl_->lua, config);
  auto result = it->second.on_experiment_start(cfg);
  if (!result.valid()) {
    sol::error err = result;
    spdlog::error("Lua on_experiment_start error: {}", err.what());
  }
}

void LuaRuntime::call_on_experiment_end(const ReportWindowStats &stats_in) {
  auto it = impl_->experiment_fns.find(impl_->selected);
  if (it == impl_->experiment_fns.end() ||
      !it->second.flags.has_on_experiment_end)
    return;

  auto &lua = impl_->lua;
  sol::table stats = lua.create_table();
  stats["step"] = stats_in.step;
  stats["window_index"] = stats_in.window_index;
  stats["best_fitness"] = stats_in.best_fitness;
  stats["avg_fitness"] = stats_in.avg_fitness;
  stats["num_species"] = stats_in.num_species;
  stats["alive_predators"] = stats_in.alive_predators;
  stats["alive_prey"] = stats_in.alive_prey;
  stats["avg_complexity"] = stats_in.avg_complexity;

  auto result = it->second.on_experiment_end(stats);
  if (!result.valid()) {
    sol::error err = result;
    spdlog::error("Lua on_experiment_end error: {}", err.what());
  }
}

} // namespace moonai
