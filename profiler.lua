local profiler_seeds = { 41, 42, 43, 44, 45, 46 }

return {
    baseline = {
        config_path = "config.lua",
        experiment = "baseline_seed42",
        generations = 60,
        output_dir = "output/profiles",
        seeds = profiler_seeds,
    },
}
