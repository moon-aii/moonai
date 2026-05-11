#include "core/lua_runtime.hpp"

#include <spdlog/spdlog.h>

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace moonai {

// ── Lua table helpers (duplicated from config.cpp — anonymous namespace) ──

namespace {

template <typename T> void lua_get(const sol::table &tbl, const char *key, T &field) {
  auto val = tbl[key];
  if (val.valid()) {
    field = val.get<T>();
  }
}

SimulationConfig table_to_config(const sol::table &tbl) {
  SimulationConfig config;

  lua_get(tbl, "grid_size", config.grid_size);
  lua_get(tbl, "predator_count", config.predator_count);
  lua_get(tbl, "prey_count", config.prey_count);
  lua_get(tbl, "predator_speed", config.predator_speed);
  lua_get(tbl, "prey_speed", config.prey_speed);
  lua_get(tbl, "vision_range", config.vision_range);
  lua_get(tbl, "interaction_range", config.interaction_range);
  lua_get(tbl, "initial_energy", config.initial_energy);
  lua_get(tbl, "max_energy", config.max_energy);
  lua_get(tbl, "energy_drain_per_step", config.energy_drain_per_step);
  lua_get(tbl, "energy_gain_from_kill", config.energy_gain_from_kill);
  lua_get(tbl, "energy_gain_from_food", config.energy_gain_from_food);
  lua_get(tbl, "food_count", config.food_count);
  lua_get(tbl, "food_respawn_rate", config.food_respawn_rate);
  lua_get(tbl, "mutation_rate", config.mutation_rate);
  lua_get(tbl, "weight_mutation_power", config.weight_mutation_power);
  lua_get(tbl, "add_node_rate", config.add_node_rate);
  lua_get(tbl, "add_connection_rate", config.add_connection_rate);
  lua_get(tbl, "delete_connection_rate", config.delete_connection_rate);
  lua_get(tbl, "max_hidden_nodes", config.max_hidden_nodes);
  lua_get(tbl, "max_age", config.max_age);
  lua_get(tbl, "max_steps", config.max_steps);
  lua_get(tbl, "compatibility_threshold", config.compatibility_threshold);
  lua_get(tbl, "compatibility_min_normalization", config.compatibility_min_normalization);
  lua_get(tbl, "c1_excess", config.c1_excess);
  lua_get(tbl, "c2_disjoint", config.c2_disjoint);
  lua_get(tbl, "c3_weight", config.c3_weight);
  lua_get(tbl, "seed", config.seed);
  lua_get(tbl, "output_dir", config.output_dir);
  lua_get(tbl, "report_interval_steps", config.report_interval_steps);
  lua_get(tbl, "mate_range", config.mate_range);
  lua_get(tbl, "reproduction_energy_threshold", config.reproduction_energy_threshold);
  lua_get(tbl, "reproduction_energy_cost", config.reproduction_energy_cost);
  lua_get(tbl, "offspring_initial_energy", config.offspring_initial_energy);
  return config;
}

void inject_defaults(sol::state &lua) {
  SimulationConfig d;
  sol::table t = lua.create_table();
  t["grid_size"] = d.grid_size;
  t["predator_count"] = d.predator_count;
  t["prey_count"] = d.prey_count;
  t["predator_speed"] = d.predator_speed;
  t["prey_speed"] = d.prey_speed;
  t["vision_range"] = d.vision_range;
  t["interaction_range"] = d.interaction_range;
  t["initial_energy"] = d.initial_energy;
  t["max_energy"] = d.max_energy;
  t["energy_drain_per_step"] = d.energy_drain_per_step;
  t["energy_gain_from_kill"] = d.energy_gain_from_kill;
  t["energy_gain_from_food"] = d.energy_gain_from_food;
  t["food_count"] = d.food_count;
  t["food_respawn_rate"] = d.food_respawn_rate;
  t["mutation_rate"] = d.mutation_rate;
  t["weight_mutation_power"] = d.weight_mutation_power;
  t["add_node_rate"] = d.add_node_rate;
  t["add_connection_rate"] = d.add_connection_rate;
  t["delete_connection_rate"] = d.delete_connection_rate;
  t["max_hidden_nodes"] = d.max_hidden_nodes;
  t["max_age"] = d.max_age;
  t["max_steps"] = d.max_steps;
  t["compatibility_threshold"] = d.compatibility_threshold;
  t["compatibility_min_normalization"] = d.compatibility_min_normalization;
  t["c1_excess"] = d.c1_excess;
  t["c2_disjoint"] = d.c2_disjoint;
  t["c3_weight"] = d.c3_weight;
  t["seed"] = d.seed;
  t["output_dir"] = d.output_dir;
  t["report_interval_steps"] = d.report_interval_steps;
  t["mate_range"] = d.mate_range;
  t["reproduction_energy_threshold"] = d.reproduction_energy_threshold;
  t["reproduction_energy_cost"] = d.reproduction_energy_cost;
  t["offspring_initial_energy"] = d.offspring_initial_energy;
  lua["moonai_defaults"] = t;
}

} // anonymous namespace

std::map<std::string, SimulationConfig> load_all_configs_lua(const std::string &filepath) {
  LuaRuntime runtime;
  return runtime.load_config(filepath);
}

// ── PIMPL ────────────────────────────────────────────────────────────────────

struct LuaRuntime::Impl {
  sol::state lua;
};

// ── Lifecycle ────────────────────────────────────────────────────────────────

LuaRuntime::LuaRuntime() : impl_(std::make_unique<Impl>()) {}

LuaRuntime::~LuaRuntime() = default;

// ── Config loading ───────────────────────────────────────────────────────────

std::map<std::string, SimulationConfig> LuaRuntime::load_config(const std::string &filepath) {
  std::map<std::string, SimulationConfig> configs;

  auto &lua = impl_->lua;
  lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::table, sol::lib::string);
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
      if (key.get_type() != sol::type::string || val.get_type() != sol::type::table)
        continue;

      std::string name = key.as<std::string>();
      sol::table exp_tbl = val.as<sol::table>();

      configs[name] = table_to_config(exp_tbl);
    }

    if (configs.empty()) {
      spdlog::error("Lua config '{}' returned no named experiments.", filepath);
    } else {
      spdlog::info("Loaded {} experiment(s) from '{}'.", configs.size(), filepath);
    }

  } catch (const std::exception &e) {
    spdlog::error("Failed to load Lua config '{}': {}", filepath, e.what());
  }

  return configs;
}
} // namespace moonai
