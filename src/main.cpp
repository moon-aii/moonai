#include "core/config.hpp"
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

    // ── Load and validate configuration ─────────────────────────────────
    auto config = moonai::load_config(args.config_path);

    // Apply CLI overrides
    if (args.seed_override != 0) config.seed = args.seed_override;
    if (args.max_generations_override != 0) config.max_generations = args.max_generations_override;

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
    moonai::SimulationManager simulation(config);

    constexpr int NN_INPUTS = moonai::SensorInput::SIZE;  // 15 sensor inputs
    constexpr int NN_OUTPUTS = 2;  // dx, dy movement direction

    moonai::EvolutionManager evolution(config, rng);

    bool resumed = false;
    if (!args.resume_path.empty()) {
        if (evolution.load_checkpoint(args.resume_path, rng)) {
            resumed = true;
        } else {
            spdlog::error("Failed to load checkpoint '{}'. Exiting.", args.resume_path);
            return 1;
        }
    }

    if (!resumed) {
        simulation.initialize();
        evolution.initialize(NN_INPUTS, NN_OUTPUTS);
    }

    // ── CUDA initialization ─────────────────────────────────────────────
    // Must happen after evolution.initialize() / load_checkpoint() so that
    // population size and num_inputs are known for GpuBatch allocation.
#ifdef MOONAI_ENABLE_CUDA
    if (!args.no_gpu && moonai::gpu::init_cuda()) {
        spdlog::info("CUDA initialized. GPU acceleration enabled.");
        moonai::gpu::print_device_info();
        evolution.enable_gpu(true);
    } else {
        spdlog::warn("CUDA unavailable or --no-gpu set. Using CPU path.");
    }
#endif

    moonai::Logger logger(config.output_dir, config.seed);
    logger.initialize(config);

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
    if (!headless) {
        visualization.initialize();
    }

    // ── Signal handling ─────────────────────────────────────────────────
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ── Main loop ───────────────────────────────────────────────────────
    spdlog::info("Starting simulation: {} predators, {} prey, {} ticks/gen",
                 config.predator_count, config.prey_count, config.generation_ticks);

    int generation = resumed ? evolution.generation() : 0;
    float dt = 1.0f / static_cast<float>(config.target_fps);

    // Set up per-tick logging callback for headless/fast-forward path
    if (config.tick_log_enabled) {
        evolution.set_tick_callback([&](int tick, const moonai::SimulationManager& sim) {
            if (tick % config.tick_log_interval == 0) {
                logger.log_tick(generation, tick, sim.agents());
            }
            logger.log_events(generation, tick, sim.last_events());
        });
    }
    int max_gen = (args.max_generations_override != 0) ? args.max_generations_override
                                                        : config.max_generations;

    while (g_running) {
        if (config.max_generations > 0 && generation >= config.max_generations) {
            break;
        }

        // Reset simulation for this generation
        simulation.reset();

        // Assign species IDs to agents for visualization coloring
        evolution.assign_species_ids(simulation);

        if (headless || visualization.is_fast_forward()) {
            // ── Headless / fast-forward mode: run generation at max speed ──
            evolution.evaluate_generation(simulation);
        } else {
            // ── Visual mode: tick-by-tick with rendering ─────────────
            const auto& networks = evolution.networks();
            int tick = 0;

            while (tick < config.generation_ticks && g_running) {
                visualization.handle_events();
                if (visualization.should_close()) {
                    g_running = 0;
                    break;
                }

                if (visualization.should_reset()) {
                    visualization.clear_reset();
                    simulation.reset();
                    tick = 0;
                    continue;
                }

                if (visualization.is_paused() && !visualization.should_step()) {
                    visualization.set_generation(generation);
                    visualization.render(simulation, evolution);
                    continue;
                }
                visualization.clear_step();

                int steps = std::min(visualization.speed_multiplier(),
                                     config.generation_ticks - tick);
                for (int s = 0; s < steps; ++s) {
                    for (size_t i = 0; i < simulation.agents().size() && i < networks.size(); ++i) {
                        if (!simulation.agents()[i]->alive()) continue;

                        auto sensors = simulation.get_sensors(i);
                        auto output = networks[i]->activate(sensors.to_vector());

                        moonai::Vec2 direction{0.0f, 0.0f};
                        if (output.size() >= 2) {
                            direction.x = output[0] * 2.0f - 1.0f;
                            direction.y = output[1] * 2.0f - 1.0f;
                        }
                        simulation.apply_action(i, direction, dt);
                    }
                    simulation.tick(dt);
                    ++tick;

                    // Per-tick logging (visual path)
                    if (config.tick_log_enabled) {
                        if (tick % config.tick_log_interval == 0) {
                            logger.log_tick(generation, tick, simulation.agents());
                        }
                        logger.log_events(generation, tick, simulation.last_events());
                    }

                    if (simulation.alive_prey() == 0 || simulation.alive_predators() == 0) break;
                    if (tick >= config.generation_ticks) break;
                }

                // Update activation values for the selected agent's NN panel
                int sel = visualization.selected_agent();
                if (sel >= 0 && sel < static_cast<int>(networks.size())
                        && simulation.agents()[sel]->alive()) {
                    visualization.set_selected_activations(
                        networks[sel]->last_activations(),
                        networks[sel]->node_index_map());
                }

                visualization.set_generation(generation);
                visualization.render(simulation, evolution);
            }

            // Compute fitness using the single authoritative formula
            evolution.compute_fitness(simulation);
        }

        // Collect metrics
        auto m = metrics.collect(generation,
                                  evolution.population(),
                                  simulation.alive_predators(),
                                  simulation.alive_prey());
        m.num_species = static_cast<int>(evolution.species().size());

        if (!headless) {
            visualization.set_fitness(m.best_fitness, m.avg_fitness);
            visualization.set_species_count(m.num_species);
            visualization.push_fitness(m.best_fitness, m.avg_fitness);
        }

        // Log
        if (generation % config.log_interval == 0) {
            logger.log_generation(m.generation, m.predator_count, m.prey_count,
                                  m.best_fitness, m.avg_fitness,
                                  m.num_species, m.avg_genome_complexity);

            const auto& pop = evolution.population();
            if (!pop.empty()) {
                auto best = std::max_element(pop.begin(), pop.end(),
                    [](const moonai::Genome& a, const moonai::Genome& b) {
                        return a.fitness() < b.fitness();
                    });
                logger.log_best_genome(generation, *best);
            }
            logger.log_species(generation, evolution.species());
            logger.flush();
        }

        spdlog::info("Gen {:>4d}: best={:.3f} avg={:.3f} species={} alive=({}/{}) complexity={:.1f}",
                     generation, m.best_fitness, m.avg_fitness,
                     m.num_species,
                     simulation.alive_predators(), simulation.alive_prey(),
                     m.avg_genome_complexity);

        // Headless progress bar
        if (headless && max_gen > 0) {
            int pct = (generation * 100) / max_gen;
            std::printf("\r  Gen %d/%d [%d%%]  best=%.3f  avg=%.3f  species=%d   ",
                        generation, max_gen, pct,
                        m.best_fitness, m.avg_fitness,
                        evolution.species_count());
            std::fflush(stdout);
        }

        // Evolve for next generation
        evolution.evolve();
        ++generation;

        // Save checkpoint if requested
        if (args.checkpoint_interval > 0 && generation % args.checkpoint_interval == 0) {
            std::string ckpt_path = config.output_dir + "/checkpoint_gen_"
                + std::to_string(generation) + ".json";
            evolution.save_checkpoint(ckpt_path, rng);
        }
    }

    if (headless && max_gen > 0) {
        std::printf("\n");
    }

    // ── Shutdown ────────────────────────────────────────────────────────
    spdlog::info("Simulation ended after {} generations.", generation);
    spdlog::info("Output saved to: {}", logger.run_dir());
    return 0;
}
