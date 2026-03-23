-- MoonAI simulation config
--
-- moonai_defaults is injected by the runtime and always reflects the C++ SimulationConfig
-- struct defaults, so this file never needs updating when parameters are added or renamed.
--
-- Usage:
--   ./moonai                                                # GUI, runs 'default' directly
--   ./moonai config.lua --list                             # list all experiments
--   ./moonai config.lua --all --headless                   # run full experiment matrix
--   ./moonai config.lua --experiment baseline_seed42       # one specific experiment

-- Shallow-copy a table and apply any number of override tables (right-most wins).
local function extend(t, ...)
    local r = {}
    for k, v in pairs(t) do r[k] = v end
    for _, overrides in ipairs({...}) do
        for k, v in pairs(overrides) do r[k] = v end
    end
    return r
end

-- Scale world size and food count proportionally to population, maintaining agent density.
-- Returns a partial table to merge via extend().
local function scale_base(pred, prey)
    local total = pred + prey
    local default_total = moonai_defaults.predator_count + moonai_defaults.prey_count
    local factor = math.sqrt(total / default_total)
    return {
        predator_count = pred,
        prey_count     = prey,
        grid_width     = math.floor(moonai_defaults.grid_width * factor),
        grid_height    = math.floor(moonai_defaults.grid_height * factor),
        food_count     = math.floor(moonai_defaults.food_count * (total / default_total)),
    }
end

-- ── Experiments ───────────────────────────────────────────────────────────────
-- All experiments start from moonai_defaults (4300×2400, 500 predators, 1500 prey,
-- 1500 ticks/gen) and override exactly the variable(s) under study.
-- 66 conditions × 5 seeds = 330 deterministic runs.

-- Pre-compute scale bases for commonly used population sizes
local base_1k  = scale_base(250,  750)
local base_3k  = scale_base(750,  2250)
local base_5k  = scale_base(1250, 3750)
local base_8k  = scale_base(2000, 6000)
local base_10k = scale_base(2500, 7500)
local base_15k = scale_base(3750, 11250)
local base_20k = scale_base(5000, 15000)

local conditions = {
    -- ── Group A: Baseline sweeps (2K agents, default world) ──────────────
    baseline       = moonai_defaults,
    mut_low        = extend(moonai_defaults, { mutation_rate = 0.1 }),
    mut_high       = extend(moonai_defaults, { mutation_rate = 0.5 }),
    mut_very_low   = extend(moonai_defaults, { mutation_rate = 0.05 }),
    mut_very_high  = extend(moonai_defaults, { mutation_rate = 0.8 }),
    pop_small      = extend(moonai_defaults, scale_base(100, 300)),
    pop_medium     = extend(moonai_defaults, scale_base(250, 750)),
    pop_large      = extend(moonai_defaults, base_5k),
    pop_huge       = extend(moonai_defaults, base_10k),
    pop_massive    = extend(moonai_defaults, base_20k),
    no_speciation  = extend(moonai_defaults, { compatibility_threshold = 100.0 }),
    tight_speciation = extend(moonai_defaults, { compatibility_threshold = 1.0 }),
    tanh           = extend(moonai_defaults, { activation_function = "tanh" }),
    relu           = extend(moonai_defaults, { activation_function = "relu" }),
    crossover_low  = extend(moonai_defaults, { crossover_rate = 0.25 }),
    crossover_none = extend(moonai_defaults, { crossover_rate = 0.0 }),

    -- ── Group B: Scale experiments (proportional world) ──────────────────
    scale_1k       = extend(moonai_defaults, base_1k),
    scale_3k       = extend(moonai_defaults, base_3k),
    scale_5k       = extend(moonai_defaults, base_5k),
    scale_8k       = extend(moonai_defaults, base_8k),
    scale_10k      = extend(moonai_defaults, base_10k),
    scale_15k      = extend(moonai_defaults, base_15k),
    scale_20k      = extend(moonai_defaults, base_20k),

    -- ── Group C: Parameter sweeps at 5K ──────────────────────────────────
    s5k_mut_low       = extend(moonai_defaults, base_5k, { mutation_rate = 0.1 }),
    s5k_mut_high      = extend(moonai_defaults, base_5k, { mutation_rate = 0.5 }),
    s5k_mut_very_high = extend(moonai_defaults, base_5k, { mutation_rate = 0.8 }),
    s5k_tanh          = extend(moonai_defaults, base_5k, { activation_function = "tanh" }),
    s5k_relu          = extend(moonai_defaults, base_5k, { activation_function = "relu" }),
    s5k_no_spec       = extend(moonai_defaults, base_5k, { compatibility_threshold = 100.0 }),
    s5k_tight_spec    = extend(moonai_defaults, base_5k, { compatibility_threshold = 1.0 }),
    s5k_crossover_low = extend(moonai_defaults, base_5k, { crossover_rate = 0.25 }),
    s5k_crossover_none= extend(moonai_defaults, base_5k, { crossover_rate = 0.0 }),

    -- ── Group D: Parameter sweeps at 10K ─────────────────────────────────
    s10k_mut_low      = extend(moonai_defaults, base_10k, { mutation_rate = 0.1 }),
    s10k_mut_high     = extend(moonai_defaults, base_10k, { mutation_rate = 0.5 }),
    s10k_tanh         = extend(moonai_defaults, base_10k, { activation_function = "tanh" }),
    s10k_relu         = extend(moonai_defaults, base_10k, { activation_function = "relu" }),
    s10k_no_spec      = extend(moonai_defaults, base_10k, { compatibility_threshold = 100.0 }),
    s10k_crossover_low= extend(moonai_defaults, base_10k, { crossover_rate = 0.25 }),

    -- ── Group E: World density (5K agents, varying world size) ───────────
    dense_5k  = extend(moonai_defaults, { predator_count = 1250, prey_count = 3750,
                    grid_width = 3000,  grid_height = 1700, food_count = 6250 }),
    normal_5k = extend(moonai_defaults, base_5k),
    sparse_5k = extend(moonai_defaults, { predator_count = 1250, prey_count = 3750,
                    grid_width = 12000, grid_height = 6750, food_count = 6250 }),
    vast_5k   = extend(moonai_defaults, { predator_count = 1250, prey_count = 3750,
                    grid_width = 15000, grid_height = 8400, food_count = 6250 }),

    -- ── Group F: Generation length ───────────────────────────────────────
    ticks_500_2k  = extend(moonai_defaults, { generation_ticks = 500 }),
    ticks_2000_2k = extend(moonai_defaults, { generation_ticks = 2000 }),
    ticks_3000_2k = extend(moonai_defaults, { generation_ticks = 3000 }),
    ticks_500_5k  = extend(moonai_defaults, base_5k, { generation_ticks = 500 }),
    ticks_2000_5k = extend(moonai_defaults, base_5k, { generation_ticks = 2000 }),
    ticks_3000_5k = extend(moonai_defaults, base_5k, { generation_ticks = 3000 }),

    -- ── Group G: Energy / resource dynamics ──────────────────────────────
    energy_scarce_2k   = extend(moonai_defaults, { initial_energy = 75.0, food_respawn_rate = 0.01 }),
    energy_abundant_2k = extend(moonai_defaults, { initial_energy = 300.0, food_respawn_rate = 0.05 }),
    energy_scarce_5k   = extend(moonai_defaults, base_5k, { initial_energy = 75.0, food_respawn_rate = 0.01 }),
    energy_abundant_5k = extend(moonai_defaults, base_5k, { initial_energy = 300.0, food_respawn_rate = 0.05 }),
    energy_extreme_5k  = extend(moonai_defaults, base_5k, { initial_energy = 50.0, food_respawn_rate = 0.005, energy_drain_per_tick = 0.15 }),
    energy_rich_5k     = extend(moonai_defaults, base_5k, { initial_energy = 500.0, food_respawn_rate = 0.08, energy_drain_per_tick = 0.03 }),

    -- ── Group H: Agent speed / interaction range (5K) ────────────────────
    fast_agents_5k   = extend(moonai_defaults, base_5k, { predator_speed = 6.0, prey_speed = 7.0 }),
    slow_agents_5k   = extend(moonai_defaults, base_5k, { predator_speed = 2.5, prey_speed = 3.0 }),
    wide_vision_5k   = extend(moonai_defaults, base_5k, { vision_range = 300.0 }),
    narrow_vision_5k = extend(moonai_defaults, base_5k, { vision_range = 80.0 }),
    long_attack_5k   = extend(moonai_defaults, base_5k, { attack_range = 40.0 }),
    short_attack_5k  = extend(moonai_defaults, base_5k, { attack_range = 10.0 }),

    -- ── Group I: Topology complexity ─────────────────────────────────────
    high_complexity_5k  = extend(moonai_defaults, base_5k, { add_node_rate = 0.1, add_connection_rate = 0.15 }),
    low_complexity_5k   = extend(moonai_defaults, base_5k, { add_node_rate = 0.01, add_connection_rate = 0.02 }),
    no_growth_5k        = extend(moonai_defaults, base_5k, { add_node_rate = 0.0, add_connection_rate = 0.0 }),
    high_complexity_10k = extend(moonai_defaults, base_10k, { add_node_rate = 0.1, add_connection_rate = 0.15 }),
    max_hidden_small_5k = extend(moonai_defaults, base_5k, { max_hidden_nodes = 10 }),
    max_hidden_large_5k = extend(moonai_defaults, base_5k, { max_hidden_nodes = 50 }),
}

local seeds = { 42, 43, 44, 45, 46 }

local experiments = {}
for name, cfg in pairs(conditions) do
    for _, seed in ipairs(seeds) do
        experiments[name .. "_seed" .. seed] = extend(cfg, {
            seed            = seed,
            max_generations = 200,
        })
    end
end

-- ── Default run ───────────────────────────────────────────────────────────────
-- Single named entry for casual use: 'just run' auto-selects this because it is
-- the only entry with this name.  All values come from moonai_defaults (2K agents).
--
-- Optional Lua callbacks:
--   fitness_fn(stats, weights) → number     Custom fitness formula (replaces C++ default)
--   on_generation_end(gen, stats) → table?  Called after each generation; return overrides or nil
--   on_experiment_start(config)             Called once before the main loop
--   on_experiment_end(stats)                Called once after the main loop
experiments["default"] = extend(moonai_defaults, {
    -- Example fitness function (mirrors the default C++ formula).
    -- Remove or comment out to use the built-in C++ default.
    fitness_fn = function(stats, weights)
        return weights.survival * stats.age_ratio
             + weights.kill     * stats.kills_or_food
             + weights.energy   * stats.energy_ratio
             + stats.alive_bonus
             + weights.distance * stats.dist_ratio
             - weights.complexity_penalty * stats.complexity
    end,

    -- Example generation hook: boost mutation when average fitness stagnates.
    -- on_generation_end = function(gen, stats)
    --     if stats.avg_fitness < 2.0 and gen > 20 then
    --         return { mutation_rate = 0.5 }
    --     end
    --     return nil
    -- end,
})

return experiments
