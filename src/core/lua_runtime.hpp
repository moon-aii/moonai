#pragma once

#include "core/config.hpp"

#include <map>
#include <memory>
#include <string>

namespace moonai {

// Flags indicating which Lua callbacks an experiment defines
struct LuaCallbacks {
    bool has_fitness_fn = false;
    bool has_on_generation_end = false;
    bool has_on_experiment_start = false;
    bool has_on_experiment_end = false;
};

// Stats passed to generation-level hooks
struct GenerationStats {
    int generation;
    float best_fitness;
    float avg_fitness;
    int num_species;
    int alive_predators;
    int alive_prey;
    float avg_complexity;
};

// Persistent Lua runtime for config loading + runtime callbacks.
// Owns the sol::state so Lua functions survive past config parsing.
// Uses PIMPL to keep sol2 out of the header.
class LuaRuntime {
public:
    LuaRuntime();
    ~LuaRuntime();

    LuaRuntime(const LuaRuntime&) = delete;
    LuaRuntime& operator=(const LuaRuntime&) = delete;

    // Load config file. Returns experiment name -> SimulationConfig.
    // Also extracts Lua callback functions from each experiment table.
    std::map<std::string, SimulationConfig> load_config(const std::string& filepath);

    // Activate one experiment's callbacks for runtime use
    void select_experiment(const std::string& name);

    // Query which callbacks the selected experiment defines
    const LuaCallbacks& callbacks() const;

    // ── Lua fitness function ──────────────────────────────────────────
    // Returns fitness computed by Lua, or -1.0f if no fitness_fn defined.
    float call_fitness(float age_ratio, float kills_or_food, float energy_ratio,
                       float alive_bonus, float dist_ratio, float complexity,
                       const SimulationConfig& config);

    // ── Event hooks ───────────────────────────────────────────────────
    // Returns true if the hook returned config overrides; fills overrides map.
    bool call_on_generation_end(const GenerationStats& stats,
                                std::map<std::string, float>& overrides);
    void call_on_experiment_start(const SimulationConfig& config);
    void call_on_experiment_end(const GenerationStats& stats);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace moonai
