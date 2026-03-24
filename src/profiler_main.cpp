#include "core/config.hpp"
#include "core/lua_runtime.hpp"
#include "core/profiler.hpp"
#include "core/profiler_suite.hpp"
#include "core/random.hpp"
#include "simulation/simulation_manager.hpp"
#include "simulation/physics.hpp"
#include "evolution/evolution_manager.hpp"
#include "evolution/neural_network.hpp"
#include "data/logger.hpp"
#include "data/metrics.hpp"

#ifdef MOONAI_ENABLE_CUDA
namespace moonai::gpu {
    bool init_cuda();
    void print_device_info();
}
#endif

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <map>

namespace {

volatile std::sig_atomic_t g_running = 1;

void signal_handler(int) {
    g_running = 0;
}

struct ProfilerCliArgs {
    std::string config_path = "profiler.lua";
    std::string suite_name;
    std::string output_dir;
    bool list_suites = false;
    bool validate_only = false;
    bool verbose = false;
    bool no_gpu = false;
};

struct RawRunSummary {
    std::uint64_t seed = 0;
    std::string run_dir;
    std::string profile_path;
    std::string config_fingerprint;
    std::string experiment_name;
    std::string base_experiment_name;
    double run_total_ms = 0.0;
    double avg_generation_ms = 0.0;
    int generation_count = 0;
    int cpu_generation_count = 0;
    int gpu_generation_count = 0;
    nlohmann::json summary_events;
    nlohmann::json summary_counters;
};

ProfilerCliArgs parse_args(int argc, char* argv[]) {
    ProfilerCliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::printf(
                "MoonAI Profiler\n\n"
                "Usage: %s [OPTIONS] [profiler.lua]\n\n"
                "Options:\n"
                "  --suite <name>         Select profiler suite\n"
                "  --list                 List profiler suites and exit\n"
                "  --validate             Validate profiler suite and exit\n"
                "  --output <dir>         Override profiler output root\n"
                "  --no-gpu               Disable CUDA GPU acceleration\n"
                "  -v, --verbose          Enable debug logging\n"
                "  -h, --help             Show this help message\n",
                argv[0]);
            std::exit(0);
        } else if (arg == "--suite" && i + 1 < argc) {
            args.suite_name = argv[++i];
        } else if (arg == "--list") {
            args.list_suites = true;
        } else if (arg == "--validate") {
            args.validate_only = true;
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--no-gpu") {
            args.no_gpu = true;
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (!arg.empty() && arg[0] != '-') {
            args.config_path = arg;
        } else {
            spdlog::warn("Unknown argument: {}", arg);
        }
    }
    return args;
}

std::string sanitize_path_component(const std::string& value) {
    std::string sanitized;
    sanitized.reserve(value.size());
    for (char ch : value) {
        const bool valid = (ch >= 'a' && ch <= 'z')
            || (ch >= 'A' && ch <= 'Z')
            || (ch >= '0' && ch <= '9')
            || ch == '-'
            || ch == '_';
        sanitized.push_back(valid ? ch : '_');
    }
    return sanitized.empty() ? std::string{"profile_suite"} : sanitized;
}

std::string utc_timestamp_for_path(std::chrono::system_clock::time_point now) {
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string utc_timestamp_iso(std::chrono::system_clock::time_point now) {
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::filesystem::path create_suite_output_dir(const std::string& root, const std::string& suite_name) {
    const auto now = std::chrono::system_clock::now();
    const std::string base_name = utc_timestamp_for_path(now) + "_" + sanitize_path_component(suite_name);
    std::filesystem::path candidate = std::filesystem::path(root) / base_name;
    for (int suffix = 2; std::filesystem::exists(candidate); ++suffix) {
        candidate = std::filesystem::path(root) / (base_name + "_" + std::to_string(suffix));
    }
    std::filesystem::create_directories(candidate / "raw");
    return candidate;
}

RawRunSummary load_raw_run_summary(const std::filesystem::path& profile_path) {
    std::ifstream handle(profile_path);
    if (!handle.is_open()) {
        throw std::runtime_error("failed to open raw profile output: " + profile_path.string());
    }

    const nlohmann::json payload = nlohmann::json::parse(handle);
    if (payload.value("schema_version", 0) < 2) {
        throw std::runtime_error("unsupported raw profile schema in " + profile_path.string());
    }

    RawRunSummary summary;
    const auto& run = payload.at("run");
    const auto& suite_summary = payload.at("summary");
    summary.seed = run.at("seed").get<std::uint64_t>();
    summary.run_dir = run.at("output_dir").get<std::string>();
    summary.profile_path = profile_path.string();
    summary.config_fingerprint = run.value("config_fingerprint", std::string{});
    summary.experiment_name = run.value("experiment_name", std::string{});
    summary.base_experiment_name = run.value("base_experiment_name", std::string{});
    summary.run_total_ms = suite_summary.value("run_total_ms", 0.0);
    summary.generation_count = suite_summary.value("generation_count", 0);
    summary.cpu_generation_count = suite_summary.value("cpu_generation_count", 0);
    summary.gpu_generation_count = suite_summary.value("gpu_generation_count", 0);
    summary.summary_events = suite_summary.at("events");
    summary.summary_counters = suite_summary.at("counters");
    summary.avg_generation_ms = summary.summary_events.at("generation_total").at("avg_ms_per_generation").get<double>();
    return summary;
}

double compute_stddev(const std::vector<double>& values, double mean) {
    if (values.empty()) {
        return 0.0;
    }
    double accum = 0.0;
    for (double value : values) {
        const double diff = value - mean;
        accum += diff * diff;
    }
    return std::sqrt(accum / static_cast<double>(values.size()));
}

RawRunSummary run_profiled_experiment(const moonai::ProfilerSuiteConfig& suite,
                                      const moonai::SimulationConfig& base_config,
                                      moonai::LuaRuntime& lua_runtime,
                                      std::uint64_t seed,
                                      const std::filesystem::path& suite_dir,
                                      bool no_gpu,
                                      const std::string& config_fingerprint) {
    moonai::SimulationConfig config = base_config;
    config.seed = seed;
    config.max_generations = suite.generations;

    auto errors = moonai::validate_config(config);
    if (!errors.empty()) {
        std::ostringstream oss;
        oss << "invalid config for suite seed " << seed;
        for (const auto& error : errors) {
            oss << " [" << error.field << ": " << error.message << "]";
        }
        throw std::runtime_error(oss.str());
    }

    constexpr int kInputs = moonai::SensorInput::SIZE;
    constexpr int kOutputs = 2;

    moonai::Random rng(config.seed);
    moonai::SimulationManager simulation(config);
    moonai::EvolutionManager evolution(config, rng);
    simulation.initialize();
    evolution.initialize(kInputs, kOutputs);
    lua_runtime.select_experiment(suite.experiment_name);
    evolution.set_lua_runtime(&lua_runtime);
    lua_runtime.call_on_experiment_start(config);

#ifdef MOONAI_ENABLE_CUDA
    if (!no_gpu && moonai::gpu::init_cuda()) {
        evolution.enable_gpu(true);
    }
#else
    (void)no_gpu;
#endif

    const std::string run_name = suite.name + "_seed" + std::to_string(seed);
    moonai::Logger logger(config.output_dir, config.seed, run_name);
    logger.initialize(config);
    moonai::MetricsCollector metrics;
    int current_generation = 0;

    if (config.tick_log_enabled) {
        evolution.set_tick_callback([&](int tick, const moonai::SimulationManager& sim) {
            if (tick % config.tick_log_interval == 0) {
                logger.log_tick(current_generation, tick, sim.agents());
            }
            logger.log_events(current_generation, tick, sim.last_events());
        });
    }

    moonai::Profiler::instance().set_enabled(true);
#ifdef MOONAI_ENABLE_CUDA
    constexpr bool kCudaCompiled = true;
#else
    constexpr bool kCudaCompiled = false;
#endif
#ifdef MOONAI_OPENMP_ENABLED
    constexpr bool kOpenmpCompiled = true;
#else
    constexpr bool kOpenmpCompiled = false;
#endif
    moonai::ProfileRunSpec run_spec;
    run_spec.experiment_name = run_name;
    run_spec.output_root_dir = (suite_dir / "raw").string();
    run_spec.seed = config.seed;
    run_spec.predator_count = config.predator_count;
    run_spec.prey_count = config.prey_count;
    run_spec.food_count = config.food_count;
    run_spec.generation_ticks = config.generation_ticks;
    run_spec.gpu_allowed = !no_gpu;
    run_spec.cuda_compiled = kCudaCompiled;
    run_spec.openmp_compiled = kOpenmpCompiled;
    run_spec.suite_name = suite.name;
    run_spec.base_experiment_name = suite.experiment_name;
    run_spec.config_fingerprint = config_fingerprint;
    run_spec.profiler_entry_point = "moonai_profiler";
    moonai::Profiler::instance().start_run(run_spec);

    const auto experiment_start = std::chrono::steady_clock::now();
    for (int generation = 0; g_running && generation < config.max_generations; ++generation) {
        current_generation = generation;
        moonai::Profiler::instance().start_generation(generation);
        simulation.reset();
        evolution.assign_species_ids(simulation);
        evolution.evaluate_generation(simulation);

        auto metrics_row = metrics.collect(
            generation,
            evolution.population(),
            simulation.alive_predators(),
            simulation.alive_prey());
        metrics_row.num_species = static_cast<int>(evolution.species().size());

        if (generation % config.log_interval == 0) {
            MOONAI_PROFILE_SCOPE(moonai::ProfileEvent::Logging);
            logger.log_generation(
                metrics_row.generation,
                metrics_row.predator_count,
                metrics_row.prey_count,
                metrics_row.best_fitness,
                metrics_row.avg_fitness,
                metrics_row.num_species,
                metrics_row.avg_genome_complexity);
            const auto& population = evolution.population();
            if (!population.empty()) {
                auto best = std::max_element(
                    population.begin(),
                    population.end(),
                    [](const moonai::Genome& lhs, const moonai::Genome& rhs) {
                        return lhs.fitness() < rhs.fitness();
                    });
                logger.log_best_genome(generation, *best);
            }
            logger.log_species(generation, evolution.species());
            logger.flush();
        }

        if (lua_runtime.callbacks().has_on_generation_end) {
            moonai::GenerationStats stats{
                generation,
                metrics_row.best_fitness,
                metrics_row.avg_fitness,
                metrics_row.num_species,
                simulation.alive_predators(),
                simulation.alive_prey(),
                metrics_row.avg_genome_complexity,
            };
            std::map<std::string, float> overrides;
            if (lua_runtime.call_on_generation_end(stats, overrides)) {
                moonai::apply_overrides_float(config, overrides);
                evolution.update_config(config);
            }
        }

        evolution.evolve();
        moonai::Profiler::instance().finish_generation({
            generation,
            simulation.alive_predators(),
            simulation.alive_prey(),
            evolution.species_count(),
            metrics_row.best_fitness,
            metrics_row.avg_fitness,
            metrics_row.avg_genome_complexity,
        });
    }

    moonai::GenerationStats end_stats{config.max_generations, 0.0f, 0.0f, 0, 0, 0, 0.0f};
    lua_runtime.call_on_experiment_end(end_stats);

    const auto experiment_end = std::chrono::steady_clock::now();
    moonai::Profiler::instance().finish_run(
        std::chrono::duration_cast<std::chrono::nanoseconds>(experiment_end - experiment_start).count());

    const std::filesystem::path profile_path =
        std::filesystem::path(moonai::Profiler::instance().output_dir()) / "profile.json";
    return load_raw_run_summary(profile_path);
}

nlohmann::json aggregate_named_stats(const std::vector<nlohmann::json>& named_stats,
                                    const char* total_key,
                                    const char* avg_key,
                                    const char* nonzero_key,
                                    const char* active_avg_key) {
    nlohmann::json result = nlohmann::json::object();
    if (named_stats.empty()) {
        return result;
    }

    const auto& first = named_stats.front();
    for (auto it = first.begin(); it != first.end(); ++it) {
        const std::string name = it.key();
        std::vector<double> totals;
        std::vector<double> avgs;
        std::vector<double> active_avgs;
        double nonzero_sum = 0.0;
        totals.reserve(named_stats.size());
        avgs.reserve(named_stats.size());
        active_avgs.reserve(named_stats.size());

        for (const auto& stats : named_stats) {
            const auto& row = stats.at(name);
            totals.push_back(row.at(total_key).get<double>());
            avgs.push_back(row.at(avg_key).get<double>());
            active_avgs.push_back(row.at(active_avg_key).get<double>());
            nonzero_sum += row.at(nonzero_key).get<double>();
        }

        const double total_mean = std::accumulate(totals.begin(), totals.end(), 0.0) / totals.size();
        const double avg_mean = std::accumulate(avgs.begin(), avgs.end(), 0.0) / avgs.size();
        const double active_avg_mean = std::accumulate(active_avgs.begin(), active_avgs.end(), 0.0) / active_avgs.size();
        result[name] = {
            {total_key, total_mean},
            {avg_key, avg_mean},
            {nonzero_key, nonzero_sum / static_cast<double>(named_stats.size())},
            {active_avg_key, active_avg_mean},
            {std::string(avg_key) + "_stddev", compute_stddev(avgs, avg_mean)},
        };
    }
    return result;
}

void write_suite_manifest(const std::filesystem::path& suite_dir,
                          const moonai::ProfilerSuiteConfig& suite,
                          const moonai::SimulationConfig& base_config,
                          const std::string& config_fingerprint,
                          const std::vector<RawRunSummary>& runs,
                          const std::vector<RawRunSummary>& kept_runs) {
    if (runs.size() != 6 || kept_runs.size() != 4) {
        throw std::runtime_error("suite manifest requires exactly 6 runs with 4 kept runs");
    }

    std::vector<RawRunSummary> ordered_runs = runs;
    std::sort(ordered_runs.begin(), ordered_runs.end(), [](const RawRunSummary& lhs, const RawRunSummary& rhs) {
        return lhs.avg_generation_ms < rhs.avg_generation_ms;
    });

    std::vector<nlohmann::json> kept_events;
    std::vector<nlohmann::json> kept_counters;
    std::vector<double> kept_generation_ms;
    std::vector<double> kept_run_total_ms;
    int cpu_generation_count_sum = 0;
    int gpu_generation_count_sum = 0;
    for (const auto& run : kept_runs) {
        kept_events.push_back(run.summary_events);
        kept_counters.push_back(run.summary_counters);
        kept_generation_ms.push_back(run.avg_generation_ms);
        kept_run_total_ms.push_back(run.run_total_ms);
        cpu_generation_count_sum += run.cpu_generation_count;
        gpu_generation_count_sum += run.gpu_generation_count;
    }

    const double avg_generation_ms = std::accumulate(
        kept_generation_ms.begin(), kept_generation_ms.end(), 0.0) / kept_generation_ms.size();
    const double avg_run_total_ms = std::accumulate(
        kept_run_total_ms.begin(), kept_run_total_ms.end(), 0.0) / kept_run_total_ms.size();

    nlohmann::json manifest;
    manifest["schema_version"] = 1;
    manifest["generated_at_utc"] = utc_timestamp_iso(std::chrono::system_clock::now());
    manifest["suite"] = {
        {"name", suite.name},
        {"config_path", suite.config_path},
        {"experiment_name", suite.experiment_name},
        {"output_dir", suite.output_dir},
        {"generations", suite.generations},
        {"config_fingerprint", config_fingerprint},
        {"seed_count", suite.seeds.size()},
        {"trim_policy", {
            {"drop_fastest", 1},
            {"drop_slowest", 1},
            {"keep_count", 4},
            {"sort_key", "generation_total.avg_ms_per_generation"},
        }},
        {"base_config", moonai::config_to_json(base_config)},
    };
    manifest["runs"] = nlohmann::json::array();
    for (std::size_t index = 0; index < ordered_runs.size(); ++index) {
        const auto& run = ordered_runs[index];
        const std::filesystem::path run_dir = std::filesystem::path(run.run_dir);
        const std::filesystem::path profile_path = std::filesystem::path(run.profile_path);
        std::string disposition = "kept";
        if (index == 0) {
            disposition = "dropped_fastest";
        } else if (index + 1 == ordered_runs.size()) {
            disposition = "dropped_slowest";
        }
        manifest["runs"].push_back({
            {"seed", run.seed},
            {"run_dir", std::filesystem::relative(run_dir, suite_dir).generic_string()},
            {"profile_path", std::filesystem::relative(profile_path, suite_dir).generic_string()},
            {"avg_generation_ms", run.avg_generation_ms},
            {"run_total_ms", run.run_total_ms},
            {"generation_count", run.generation_count},
            {"disposition", disposition},
        });
    }
    manifest["aggregate"] = {
        {"kept_run_count", kept_runs.size()},
        {"avg_generation_ms", avg_generation_ms},
        {"avg_generation_ms_stddev", compute_stddev(kept_generation_ms, avg_generation_ms)},
        {"avg_run_total_ms", avg_run_total_ms},
        {"cpu_generation_count_avg", static_cast<double>(cpu_generation_count_sum) / kept_runs.size()},
        {"gpu_generation_count_avg", static_cast<double>(gpu_generation_count_sum) / kept_runs.size()},
        {"events", aggregate_named_stats(
            kept_events,
            "total_ms",
            "avg_ms_per_generation",
            "nonzero_generation_count",
            "avg_ms_per_nonzero_generation")},
        {"counters", aggregate_named_stats(
            kept_counters,
            "total",
            "avg_per_generation",
            "nonzero_generation_count",
            "avg_per_nonzero_generation")},
    };
    const std::filesystem::path output_path = suite_dir / "profile_suite.json";
    std::ofstream handle(output_path);
    if (!handle.is_open()) {
        throw std::runtime_error("failed to open suite manifest for writing: " + output_path.string());
    }
    handle << manifest.dump(2) << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    const ProfilerCliArgs args = parse_args(argc, argv);
    spdlog::set_level(args.verbose ? spdlog::level::debug : spdlog::level::info);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    const auto suites = moonai::load_profiler_suites_lua(args.config_path);
    if (suites.empty()) {
        spdlog::error("No profiler suites loaded from '{}'", args.config_path);
        return 1;
    }

    if (args.list_suites) {
        std::printf("Profiler suites in '%s':\n", args.config_path.c_str());
        for (const auto& [name, _] : suites) {
            std::printf("  %s\n", name.c_str());
        }
        return 0;
    }

    moonai::ProfilerSuiteConfig suite;
    if (!args.suite_name.empty()) {
        auto it = suites.find(args.suite_name);
        if (it == suites.end()) {
            spdlog::error("Profiler suite '{}' not found.", args.suite_name);
            return 1;
        }
        suite = it->second;
    } else if (suites.size() == 1) {
        suite = suites.begin()->second;
    } else {
        spdlog::error("Multiple profiler suites found. Use --suite.");
        return 1;
    }

    moonai::LuaRuntime lua_runtime;
    auto all_configs = lua_runtime.load_config(suite.config_path);
    auto it = all_configs.find(suite.experiment_name);
    if (it == all_configs.end()) {
        spdlog::error("Base experiment '{}' not found in '{}'.", suite.experiment_name, suite.config_path);
        return 1;
    }

    moonai::SimulationConfig base_config = it->second;
    base_config.max_generations = suite.generations;
    const std::string config_fingerprint = moonai::fingerprint_config(base_config);

    if (suite.seeds.size() != 6) {
        spdlog::error("Profiler suite '{}' must define exactly 6 seeds.", suite.name);
        return 1;
    }

    if (args.validate_only) {
        auto errors = moonai::validate_config(base_config);
        if (!errors.empty()) {
            for (const auto& error : errors) {
                spdlog::error("Config error [{}]: {}", error.field, error.message);
            }
            return 1;
        }
        std::printf("OK\n");
        return 0;
    }

    const std::string output_root = args.output_dir.empty() ? suite.output_dir : args.output_dir;
    const std::filesystem::path suite_dir = create_suite_output_dir(output_root, suite.name);

    std::vector<RawRunSummary> runs;
    runs.reserve(suite.seeds.size());
    for (std::size_t index = 0; index < suite.seeds.size(); ++index) {
        const std::uint64_t seed = suite.seeds[index];
        spdlog::info("=== Profiler run {}/{} (seed={}) ===", index + 1, suite.seeds.size(), seed);
        try {
            runs.push_back(run_profiled_experiment(
                suite,
                base_config,
                lua_runtime,
                seed,
                suite_dir,
                args.no_gpu,
                config_fingerprint));
        } catch (const std::exception& e) {
            spdlog::error("Profiler run failed for seed {}: {}", seed, e.what());
            return 1;
        }
    }

    for (const auto& run : runs) {
        if (run.config_fingerprint != config_fingerprint) {
            spdlog::error("Run seed {} has mismatched config fingerprint.", run.seed);
            return 1;
        }
    }

    std::sort(runs.begin(), runs.end(), [](const RawRunSummary& lhs, const RawRunSummary& rhs) {
        return lhs.avg_generation_ms < rhs.avg_generation_ms;
    });
    std::vector<RawRunSummary> kept_runs(runs.begin() + 1, runs.end() - 1);

    try {
        write_suite_manifest(suite_dir, suite, base_config, config_fingerprint, runs, kept_runs);
    } catch (const std::exception& e) {
        spdlog::error("Failed to write suite manifest: {}", e.what());
        return 1;
    }

    spdlog::info("Profiler suite output: {}", suite_dir.string());
    return 0;
}
