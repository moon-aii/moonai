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
--   ./moonai config.lua --set mutation_rate=0.5            # ad-hoc override on default

-- Shallow-copy a table and apply any number of override tables (right-most wins).
local function extend(t, ...)
    local r = {}
    for k, v in pairs(t) do r[k] = v end
    for _, overrides in ipairs({...}) do
        for k, v in pairs(overrides) do r[k] = v end
    end
    return r
end

-- ── Experiments ───────────────────────────────────────────────────────────────
-- All experiments start from moonai_defaults (1920×1080, 100 predators, 300 prey,
-- 1000 ticks/gen) and override exactly the variable under study.
-- 8 conditions × 5 seeds = 40 deterministic runs.

local conditions = {
    -- Reference: unmodified defaults, used as comparison baseline for all others.
    baseline      = moonai_defaults,

    -- Mutation rate: how strongly weight perturbation drives exploration.
    mut_low       = extend(moonai_defaults, { mutation_rate = 0.1 }),
    mut_high      = extend(moonai_defaults, { mutation_rate = 0.5 }),

    -- Population size: effect of gene pool diversity and competitive pressure.
    pop_small     = extend(moonai_defaults, { predator_count = 40,  prey_count = 120 }),
    pop_large     = extend(moonai_defaults, { predator_count = 200, prey_count = 600 }),

    -- Speciation disabled: threshold so high that all genomes share one species,
    -- removing the innovation protection speciation normally provides.
    no_speciation = extend(moonai_defaults, { compatibility_threshold = 100.0 }),

    -- Activation function: tanh vs sigmoid output range and gradient shape.
    tanh          = extend(moonai_defaults, { activation_function = "tanh" }),

    -- Crossover pressure: compare sexual vs near-asexual reproduction.
    crossover_low = extend(moonai_defaults, { crossover_rate = 0.25 }),
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
-- the only entry with this name.  All values come from moonai_defaults.
experiments["default"] = moonai_defaults

return experiments
