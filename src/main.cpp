#include "core/config.hpp"
#include "core/lua_runtime.hpp"
#include "core/profiler.hpp"
#include "core/random.hpp"
#include "simulation/simulation_manager.hpp"
#include "simulation/physics.hpp"
#include "evolution/evolution_manager.hpp"
#include "evolution/neural_network.hpp"
#include "data/logger.hpp"
#include "data/metrics.hpp"

#include "visualization/visualization_manager.hpp"

#ifdef MOONAI_ENABLE_CUDA
namespace moonai::gpu {
    bool init_cuda();
    void print_device_info();
}
#endif

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <csignal>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace {
    volatile std::sig_atomic_t g_running = 1;
    void signal_handler(int) { g_running = 0; }

    std::string read_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return "";
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    const char* node_type_name(moonai::NodeType t) {
        switch (t) {
            case moonai::NodeType::Input:  return "Input";
            case moonai::NodeType::Bias:   return "Bias";
            case moonai::NodeType::Hidden: return "Hidden";
            case moonai::NodeType::Output: return "Output";
        }
        return "?";
    }

    void print_genome_diff(const moonai::Genome& a, const moonai::Genome& b) {
        // Nodes
        std::map<std::uint32_t, moonai::NodeType> nodes_a, nodes_b;
        for (const auto& n : a.nodes()) nodes_a[n.id] = n.type;
        for (const auto& n : b.nodes()) nodes_b[n.id] = n.type;

        std::printf("=== Genome Diff ===\n");
        std::printf("Nodes:  A=%d  B=%d\n",
            static_cast<int>(nodes_a.size()), static_cast<int>(nodes_b.size()));

        for (const auto& [id, t] : nodes_a) {
            if (!nodes_b.count(id))
                std::printf("  - Removed node %u (%s)\n", id, node_type_name(t));
        }
        for (const auto& [id, t] : nodes_b) {
            if (!nodes_a.count(id))
                std::printf("  + Added node %u (%s)\n", id, node_type_name(t));
        }

        // Connections by innovation
        std::map<std::uint32_t, const moonai::ConnectionGene*> conns_a, conns_b;
        for (const auto& c : a.connections()) conns_a[c.innovation] = &c;
        for (const auto& c : b.connections()) conns_b[c.innovation] = &c;

        int enabled_a = 0, enabled_b = 0;
        for (const auto& c : a.connections()) if (c.enabled) ++enabled_a;
        for (const auto& c : b.connections()) if (c.enabled) ++enabled_b;

        std::printf("Connections: A=%d (enabled %d)  B=%d (enabled %d)\n",
            static_cast<int>(conns_a.size()), enabled_a,
            static_cast<int>(conns_b.size()), enabled_b);

        for (const auto& [innov, ca] : conns_a) {
            auto it = conns_b.find(innov);
            if (it == conns_b.end()) {
                std::printf("  - Removed conn #%u (%u->%u w=%.4f%s)\n",
                    innov, ca->in_node, ca->out_node, ca->weight,
                    ca->enabled ? "" : " [disabled]");
            } else {
                const auto* cb = it->second;
                float dw = cb->weight - ca->weight;
                if (std::abs(dw) > 0.01f || ca->enabled != cb->enabled) {
                    std::printf("  ~ Conn #%u (%u->%u): w %.4f->%.4f (delta %.4f)%s\n",
                        innov, ca->in_node, ca->out_node,
                        ca->weight, cb->weight, dw,
                        (ca->enabled != cb->enabled)
                            ? (cb->enabled ? " [enabled]" : " [disabled]") : "");
                }
            }
        }
        for (const auto& [innov, cb] : conns_b) {
            if (!conns_a.count(innov)) {
                std::printf("  + Added conn #%u (%u->%u w=%.4f%s)\n",
                    innov, cb->in_node, cb->out_node, cb->weight,
                    cb->enabled ? "" : " [disabled]");
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // ── Parse CLI ───────────────────────────────────────────────────────
    auto args = moonai::parse_args(argc, argv);

    if (args.help) {
        moonai::print_usage(argv[0]);
        return 0;
    }

    // Genome diff mode: --compare <a> <b>
    // Accepts genome JSON files (output of genome.to_json()) or checkpoint JSON files.
    // For checkpoints, the best genome (population[0]) is used.
    if (!args.compare_a.empty()) {
        auto load_genome = [](const std::string& path) -> std::string {
            std::string content = read_file(path);
            if (content.empty()) return "";
            // If it looks like a checkpoint (has "population" key), extract first genome
            try {
                nlohmann::json j = nlohmann::json::parse(content);
                if (j.contains("population") && j["population"].is_array()
                        && !j["population"].empty()) {
                    return j["population"][0]["genome"].dump();
                }
            } catch (...) {}
            return content;
        };

        std::string json_a = load_genome(args.compare_a);
        std::string json_b = load_genome(args.compare_b);
        if (json_a.empty()) {
            std::fprintf(stderr, "Cannot read '%s'\n", args.compare_a.c_str());
            return 1;
        }
        if (json_b.empty()) {
            std::fprintf(stderr, "Cannot read '%s'\n", args.compare_b.c_str());
            return 1;
        }
        print_genome_diff(moonai::Genome::from_json(json_a), moonai::Genome::from_json(json_b));
        return 0;
    }

    spdlog::set_level(args.verbose ? spdlog::level::debug : spdlog::level::info);
    moonai::Profiler::instance().set_enabled(false);
    spdlog::info("MoonAI v{}.{}.{}", 0, 3, 0);
    {
        std::string features = " +vis";
#ifdef MOONAI_ENABLE_CUDA
        features += " +cuda";
#else
        features += " -cuda";
#endif
#ifdef MOONAI_OPENMP_ENABLED
        features += " +openmp";
#endif
        spdlog::info("Build features:{}", features);
    }

    // ── Load configuration from Lua ─────────────────────────────────────
    // --list and --validate use the lightweight loader (no persistent state).
    // Everything else uses LuaRuntime so Lua callbacks survive config loading.
    if (args.list_experiments) {
        auto all = moonai::load_all_configs_lua(args.config_path);
        if (all.empty()) { spdlog::error("No configs loaded from '{}'", args.config_path); return 1; }
        std::printf("Experiments in '%s':\n", args.config_path.c_str());
        for (const auto& [name, _] : all) std::printf("  %s\n", name.c_str());
        std::printf("Total: %zu\n", all.size());
        return 0;
    }

    moonai::LuaRuntime lua_runtime;
    auto all_configs = lua_runtime.load_config(args.config_path);
    if (all_configs.empty()) {
        spdlog::error("No configs loaded from '{}'", args.config_path);
        return 1;
    }

    // --all: run all experiments sequentially (headless only)
    if (args.run_all) {
        if (!args.headless) {
            spdlog::error("--all requires --headless");
            return 1;
        }
        int total = static_cast<int>(all_configs.size());
        int idx = 0;
        int failures = 0;
        for (auto& [name, cfg] : all_configs) {
            ++idx;
            spdlog::info("=== Experiment {}/{}: {} ===", idx, total, name);

            // Apply --set overrides
            if (!args.overrides.empty()) {
                auto override_errors = moonai::apply_overrides(cfg, args.overrides);
                for (const auto& e : override_errors) {
                    spdlog::error("Override error [{}]: {}", e.field, e.message);
                }
                if (!override_errors.empty()) { ++failures; continue; }
            }

            // CLI overrides
            if (args.seed_override != 0) cfg.seed = args.seed_override;
            if (args.max_generations_override != 0) cfg.max_generations = args.max_generations_override;

            auto errors = moonai::validate_config(cfg);
            if (!errors.empty()) {
                for (const auto& e : errors) {
                    spdlog::error("Config error [{}]: {}", e.field, e.message);
                }
                ++failures;
                continue;
            }

            // Seed RNG
            if (cfg.seed == 0) {
                cfg.seed = static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count());
            }
            spdlog::info("Seed: {}", cfg.seed);
            moonai::Random rng(cfg.seed);

            // Initialize subsystems
            moonai::SimulationManager simulation(cfg);
            constexpr int NN_INPUTS = moonai::SensorInput::SIZE;
            constexpr int NN_OUTPUTS = 2;
            moonai::EvolutionManager evolution(cfg, rng);
            simulation.initialize();
            evolution.initialize(NN_INPUTS, NN_OUTPUTS);

            // Wire Lua runtime for fitness functions and hooks
            lua_runtime.select_experiment(name);
            evolution.set_lua_runtime(&lua_runtime);
            lua_runtime.call_on_experiment_start(cfg);

#ifdef MOONAI_ENABLE_CUDA
            if (!args.no_gpu && moonai::gpu::init_cuda()) {
                evolution.enable_gpu(true);
            }
#endif

            moonai::Logger logger(cfg.output_dir, cfg.seed, name);
            logger.initialize(cfg);
            moonai::MetricsCollector metrics;
            if (cfg.tick_log_enabled) {
                int gen = 0;
                evolution.set_tick_callback([&](int tick, const moonai::SimulationManager& sim) {
                    if (tick % cfg.tick_log_interval == 0) {
                        logger.log_tick(gen, tick, sim.agents());
                    }
                    logger.log_events(gen, tick, sim.last_events());
                });
            }

            int gen = 0;
            while (g_running && (cfg.max_generations == 0 || gen < cfg.max_generations)) {
                simulation.reset();
                evolution.assign_species_ids(simulation);
                evolution.evaluate_generation(simulation);

                auto m = metrics.collect(gen, evolution.population(),
                                          simulation.alive_predators(), simulation.alive_prey());
                m.num_species = static_cast<int>(evolution.species().size());

                if (gen % cfg.log_interval == 0) {
                    MOONAI_PROFILE_SCOPE(moonai::ProfileEvent::Logging);
                    logger.log_generation(m.generation, m.predator_count, m.prey_count,
                                          m.best_fitness, m.avg_fitness, m.num_species,
                                          m.avg_genome_complexity);
                    const auto& pop = evolution.population();
                    if (!pop.empty()) {
                        auto best = std::max_element(pop.begin(), pop.end(),
                            [](const moonai::Genome& a, const moonai::Genome& b) {
                                return a.fitness() < b.fitness();
                            });
                        logger.log_best_genome(gen, *best);
                    }
                    logger.log_species(gen, evolution.species());
                    logger.flush();
                }

                if (cfg.max_generations > 0) {
                    int pct = (gen * 100) / cfg.max_generations;
                    std::printf("\r  [%s] Gen %d/%d [%d%%]  best=%.3f  avg=%.3f   ",
                                name.c_str(), gen, cfg.max_generations, pct,
                                m.best_fitness, m.avg_fitness);
                    std::fflush(stdout);
                }

                // Fire on_generation_end hook (may return config overrides)
                if (lua_runtime.callbacks().has_on_generation_end) {
                    moonai::GenerationStats gs{gen, m.best_fitness, m.avg_fitness,
                        m.num_species, simulation.alive_predators(),
                        simulation.alive_prey(), m.avg_genome_complexity};
                    std::map<std::string, float> overrides;
                    if (lua_runtime.call_on_generation_end(gs, overrides)) {
                        moonai::apply_overrides_float(cfg, overrides);
                        evolution.update_config(cfg);
                    }
                }

                evolution.evolve();
                ++gen;
            }
            // Fire on_experiment_end hook
            {
                moonai::GenerationStats gs{gen, 0.0f, 0.0f, 0, 0, 0, 0.0f};
                lua_runtime.call_on_experiment_end(gs);
            }
            std::printf("\n");
            spdlog::info("Output: {}", logger.run_dir());
        }
        spdlog::info("Batch complete: {}/{} succeeded", total - failures, total);
        return failures > 0 ? 1 : 0;
    }

    // ── Single experiment mode ───────────────────────────────────────────
    moonai::SimulationConfig config;
    std::string selected_experiment;

    if (!args.experiment_name.empty()) {
        // --experiment: select one from multi-config
        auto it = all_configs.find(args.experiment_name);
        if (it == all_configs.end()) {
            spdlog::error("Experiment '{}' not found. Use --list to see available experiments.",
                          args.experiment_name);
            return 1;
        }
        config = it->second;
        selected_experiment = args.experiment_name;
    } else if (all_configs.size() == 1) {
        config = all_configs.begin()->second;
        selected_experiment = all_configs.begin()->first;
    } else {
        // Multi-config without --experiment or --all: store names for GUI selector
        // For now in headless mode, pick first
        if (args.headless) {
            spdlog::warn("Multiple experiments found. Use --experiment or --all. Using first.");
            config = all_configs.begin()->second;
            selected_experiment = all_configs.begin()->first;
        } else {
            // GUI mode: will use experiment selector (handled later)
            config = all_configs.begin()->second;
            selected_experiment = all_configs.begin()->first;
        }
    }
    lua_runtime.select_experiment(selected_experiment);

    // Apply --set overrides
    if (!args.overrides.empty()) {
        auto override_errors = moonai::apply_overrides(config, args.overrides);
        for (const auto& e : override_errors) {
            spdlog::error("Override error [{}]: {}", e.field, e.message);
        }
        if (!override_errors.empty()) return 1;
    }

    // Apply CLI overrides
    if (args.seed_override != 0) config.seed = args.seed_override;
    if (args.max_generations_override != 0) config.max_generations = args.max_generations_override;

    // --validate: check config and exit
    if (args.validate_only) {
        auto errors = moonai::validate_config(config);
        if (errors.empty()) {
            std::printf("OK\n");
            return 0;
        }
        for (const auto& e : errors) {
            std::fprintf(stderr, "ERROR [%s]: %s\n", e.field.c_str(), e.message.c_str());
        }
        return 1;
    }

    auto errors = moonai::validate_config(config);
    if (!errors.empty()) {
        for (const auto& e : errors) {
            spdlog::error("Config error [{}]: {}", e.field, e.message);
        }
        return 1;
    }

    // ── Seed RNG ────────────────────────────────────────────────────────
    if (config.seed == 0) {
        config.seed = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    spdlog::info("Master seed: {}", config.seed);

    moonai::Random rng(config.seed);

    // ── Initialize subsystems ───────────────────────────────────────────
    constexpr int NN_INPUTS = moonai::SensorInput::SIZE;  // 15 sensor inputs
    constexpr int NN_OUTPUTS = 2;  // dx, dy movement direction

    auto p_simulation = std::make_unique<moonai::SimulationManager>(config);
    auto p_evolution = std::make_unique<moonai::EvolutionManager>(config, rng);

    bool resumed = false;
    if (!args.resume_path.empty()) {
        if (p_evolution->load_checkpoint(args.resume_path, rng)) {
            resumed = true;
        } else {
            spdlog::error("Failed to load checkpoint '{}'. Exiting.", args.resume_path);
            return 1;
        }
    }

    if (!resumed) {
        p_simulation->initialize();
        p_evolution->initialize(NN_INPUTS, NN_OUTPUTS);
    }

    // Wire Lua runtime for fitness functions and hooks
    p_evolution->set_lua_runtime(&lua_runtime);

    // ── CUDA initialization ─────────────────────────────────────────────
    // Must happen after evolution.initialize() / load_checkpoint() so that
    // population size and num_inputs are known for GpuBatch allocation.
#ifdef MOONAI_ENABLE_CUDA
    if (!args.no_gpu && moonai::gpu::init_cuda()) {
        spdlog::info("CUDA initialized. GPU acceleration enabled.");
        moonai::gpu::print_device_info();
        p_evolution->enable_gpu(true);
    } else {
        spdlog::warn("CUDA unavailable or --no-gpu set. Using CPU path.");
    }
#endif

    // Determine output name: --name flag, --experiment name, or empty (timestamp)
    std::string output_name = args.run_name;
    if (output_name.empty() && !args.experiment_name.empty()) {
        output_name = args.experiment_name;
    }
    auto logger = std::make_unique<moonai::Logger>(config.output_dir, config.seed, output_name);
    logger->initialize(config);

    moonai::MetricsCollector metrics;

    bool headless = args.headless;
    // Auto-detect headless environment: if no display server is available,
    // fall back to headless rather than crashing inside SFML.
    if (!headless
            && std::getenv("DISPLAY") == nullptr
            && std::getenv("WAYLAND_DISPLAY") == nullptr) {
        spdlog::warn("No display server found ($DISPLAY/$WAYLAND_DISPLAY unset). "
                     "Switching to headless mode automatically.");
        headless = true;
    }
    moonai::VisualizationManager visualization(config);
    bool has_multi_config = all_configs.size() > 1 && args.experiment_name.empty();
    if (!headless) {
        visualization.initialize();
        if (has_multi_config) {
            std::vector<std::string> exp_names;
            exp_names.reserve(all_configs.size());
            for (const auto& [name, _] : all_configs) {
                exp_names.push_back(name);
            }
            visualization.set_experiments(exp_names);
        }
    }

    // ── Signal handling ─────────────────────────────────────────────────
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ── Main loop state ─────────────────────────────────────────────
    int generation = resumed ? p_evolution->generation() : 0;
    float dt = 1.0f / static_cast<float>(config.target_fps);
    // ── Experiment selector loop (GUI multi-config) ──────────────────
    auto reinit_for_experiment = [&](const std::string& exp_name) {
        auto it = all_configs.find(exp_name);
        if (it == all_configs.end()) return false;
        config = it->second;
        if (args.seed_override != 0) config.seed = args.seed_override;
        if (args.max_generations_override != 0) config.max_generations = args.max_generations_override;
        if (!args.overrides.empty()) {
            moonai::apply_overrides(config, args.overrides);
        }
        auto errs = moonai::validate_config(config);
        if (!errs.empty()) {
            for (const auto& e : errs) spdlog::error("Config error [{}]: {}", e.field, e.message);
            return false;
        }
        if (config.seed == 0) {
            config.seed = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
        }
        rng = moonai::Random(config.seed);
        p_simulation = std::make_unique<moonai::SimulationManager>(config);
        p_evolution = std::make_unique<moonai::EvolutionManager>(config, rng);
        p_simulation->initialize();
        p_evolution->initialize(NN_INPUTS, NN_OUTPUTS);
        lua_runtime.select_experiment(exp_name);
        p_evolution->set_lua_runtime(&lua_runtime);
        lua_runtime.call_on_experiment_start(config);
#ifdef MOONAI_ENABLE_CUDA
        if (!args.no_gpu && moonai::gpu::init_cuda()) {
            p_evolution->enable_gpu(true);
        }
#endif
        output_name = exp_name;
        logger = std::make_unique<moonai::Logger>(config.output_dir, config.seed, output_name);
        logger->initialize(config);
        generation = 0;
        spdlog::info("Loaded experiment: {} (seed={})", exp_name, config.seed);
        return true;
    };

    // ── Main loop ───────────────────────────────────────────────────────
    lua_runtime.call_on_experiment_start(config);
    spdlog::info("Starting simulation: {} predators, {} prey, {} ticks/gen",
                 config.predator_count, config.prey_count, config.generation_ticks);

    // Set up per-tick logging callback for headless/fast-forward path
    if (config.tick_log_enabled) {
        p_evolution->set_tick_callback([&](int tick, const moonai::SimulationManager& sim) {
            if (tick % config.tick_log_interval == 0) {
                logger->log_tick(generation, tick, sim.agents());
            }
            logger->log_events(generation, tick, sim.last_events());
        });
    }
    int max_gen = (args.max_generations_override != 0) ? args.max_generations_override
                                                        : config.max_generations;

    while (g_running) {
        // Handle experiment selector mode
        if (!headless && visualization.in_experiment_select_mode()) {
            visualization.handle_events();
            if (visualization.should_close()) {
                g_running = 0;
                break;
            }
            visualization.render(*p_simulation, *p_evolution);
            if (visualization.experiment_was_selected()) {
                visualization.clear_experiment_selected();
                reinit_for_experiment(visualization.selected_experiment());
            }
            continue;
        }

        if (config.max_generations > 0 && generation >= config.max_generations) {
            break;
        }

        // Reset simulation for this generation
        p_simulation->reset();

        // Assign species IDs to agents for visualization coloring
        p_evolution->assign_species_ids(*p_simulation);

        if (headless || visualization.is_fast_forward()) {
            // ── Headless / fast-forward mode: run generation at max speed ──
            p_evolution->evaluate_generation(*p_simulation);
        } else {
            // ── Visual mode: tick-by-tick with rendering ─────────────
            const auto& networks = p_evolution->networks();
            int tick = 0;
            std::vector<moonai::Vec2> gpu_actions(p_simulation->agents().size(), {0.0f, 0.0f});

#ifdef MOONAI_ENABLE_CUDA
            bool visual_gpu_ready = p_evolution->prepare_gpu_generation();
#else
            bool visual_gpu_ready = false;
#endif

            while (tick < config.generation_ticks && g_running) {
                visualization.handle_events();
                if (visualization.should_close()) {
                    g_running = 0;
                    break;
                }

                // If E key was pressed, break to experiment selector
                if (visualization.in_experiment_select_mode()) break;

                if (visualization.should_reset()) {
                    visualization.clear_reset();
                    p_simulation->reset();
                    tick = 0;
                    continue;
                }

                if (visualization.is_paused() && !visualization.should_step()) {
                    visualization.set_generation(generation);
                    visualization.render(*p_simulation, *p_evolution);
                    continue;
                }
                visualization.clear_step();

                int steps = std::min(visualization.speed_multiplier(),
                                     config.generation_ticks - tick);
                for (int s = 0; s < steps; ++s) {
                    bool used_gpu = false;
#ifdef MOONAI_ENABLE_CUDA
                    if (visual_gpu_ready && p_evolution->infer_actions_gpu(*p_simulation, gpu_actions)) {
                        used_gpu = true;
                        MOONAI_PROFILE_MARK_GPU_USED(true);
                        for (size_t i = 0; i < p_simulation->agents().size() && i < networks.size(); ++i) {
                            if (!p_simulation->agents()[i]->alive()) continue;
                            p_simulation->apply_action(i, gpu_actions[i], dt);
                        }
                    } else if (visual_gpu_ready) {
                        visual_gpu_ready = false;
                    }
#endif
                    if (!used_gpu) {
                        MOONAI_PROFILE_MARK_CPU_USED(true);
                        MOONAI_PROFILE_SCOPE(moonai::ProfileEvent::CpuEvalTotal);
                        for (size_t i = 0; i < p_simulation->agents().size() && i < networks.size(); ++i) {
                            if (!p_simulation->agents()[i]->alive()) continue;

                            auto sensors = p_simulation->get_sensors(i);
                            auto output = networks[i]->activate(sensors.to_vector());

                            moonai::Vec2 direction{0.0f, 0.0f};
                            if (output.size() >= 2) {
                                direction.x = output[0] * 2.0f - 1.0f;
                                direction.y = output[1] * 2.0f - 1.0f;
                            }
                            p_simulation->apply_action(i, direction, dt);
                        }
                    }
                    p_simulation->tick(dt);
                    MOONAI_PROFILE_INC(moonai::ProfileCounter::TicksExecuted);
                    ++tick;

                    // Per-tick logging (visual path)
                    if (config.tick_log_enabled) {
                        if (tick % config.tick_log_interval == 0) {
                            logger->log_tick(generation, tick, p_simulation->agents());
                        }
                        logger->log_events(generation, tick, p_simulation->last_events());
                    }

                    if (p_simulation->alive_prey() == 0 || p_simulation->alive_predators() == 0) break;
                    if (tick >= config.generation_ticks) break;
                }

                // Update activation values for the selected agent's NN panel
                int sel = visualization.selected_agent();
                if (sel >= 0 && sel < static_cast<int>(networks.size())
                        && p_simulation->agents()[sel]->alive()) {
                    networks[sel]->activate(p_simulation->get_sensors(static_cast<size_t>(sel)).to_vector());
                    visualization.set_selected_activations(
                        networks[sel]->last_activations(),
                        networks[sel]->node_index_map());
                }

                visualization.set_generation(generation);
                visualization.render(*p_simulation, *p_evolution);
            }

            // If we broke out for experiment selector, skip the rest of the generation
            if (visualization.in_experiment_select_mode()) continue;

            // Compute fitness using the single authoritative formula
            p_evolution->compute_fitness(*p_simulation);
        }

        // Collect metrics
        auto m = metrics.collect(generation,
                                  p_evolution->population(),
                                  p_simulation->alive_predators(),
                                  p_simulation->alive_prey());
        m.num_species = static_cast<int>(p_evolution->species().size());

        if (!headless) {
            visualization.set_fitness(m.best_fitness, m.avg_fitness);
            visualization.set_species_count(m.num_species);
            visualization.push_fitness(m.best_fitness, m.avg_fitness);
        }

        // Log
        if (generation % config.log_interval == 0) {
            MOONAI_PROFILE_SCOPE(moonai::ProfileEvent::Logging);
            logger->log_generation(m.generation, m.predator_count, m.prey_count,
                                   m.best_fitness, m.avg_fitness,
                                   m.num_species, m.avg_genome_complexity);

            const auto& pop = p_evolution->population();
            if (!pop.empty()) {
                auto best = std::max_element(pop.begin(), pop.end(),
                    [](const moonai::Genome& a, const moonai::Genome& b) {
                        return a.fitness() < b.fitness();
                    });
                logger->log_best_genome(generation, *best);
            }
            logger->log_species(generation, p_evolution->species());
            logger->flush();
        }

        spdlog::info("Gen {:>4d}: best={:.3f} avg={:.3f} species={} alive=({}/{}) complexity={:.1f}",
                     generation, m.best_fitness, m.avg_fitness,
                     m.num_species,
                     p_simulation->alive_predators(), p_simulation->alive_prey(),
                     m.avg_genome_complexity);

        // Headless progress bar
        if (headless && max_gen > 0) {
            int pct = (generation * 100) / max_gen;
            std::printf("\r  Gen %d/%d [%d%%]  best=%.3f  avg=%.3f  species=%d   ",
                        generation, max_gen, pct,
                        m.best_fitness, m.avg_fitness,
                        p_evolution->species_count());
            std::fflush(stdout);
        }

        // Fire on_generation_end hook (may return config overrides)
        if (lua_runtime.callbacks().has_on_generation_end) {
            moonai::GenerationStats gs{generation, m.best_fitness, m.avg_fitness,
                m.num_species, p_simulation->alive_predators(),
                p_simulation->alive_prey(), m.avg_genome_complexity};
            std::map<std::string, float> overrides;
            if (lua_runtime.call_on_generation_end(gs, overrides)) {
                moonai::apply_overrides_float(config, overrides);
                p_evolution->update_config(config);
            }
        }

        // Evolve for next generation
        p_evolution->evolve();
        ++generation;

        // Save checkpoint if requested
        if (args.checkpoint_interval > 0 && generation % args.checkpoint_interval == 0) {
            std::string ckpt_path = config.output_dir + "/checkpoint_gen_"
                + std::to_string(generation) + ".json";
            p_evolution->save_checkpoint(ckpt_path, rng);
        }
    }

    // Fire on_experiment_end hook
    {
        moonai::GenerationStats gs{generation, 0.0f, 0.0f, 0, 0, 0, 0.0f};
        lua_runtime.call_on_experiment_end(gs);
    }

    if (headless && max_gen > 0) {
        std::printf("\n");
    }

    // ── Shutdown ────────────────────────────────────────────────────────
    spdlog::info("Simulation ended after {} generations.", generation);
    spdlog::info("Output saved to: {}", logger->run_dir());
    return 0;
}
