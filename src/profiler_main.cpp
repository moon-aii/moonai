#include "core/config.hpp"
#include "core/profiler_types.hpp"
#include "core/random.hpp"
#include "data/metrics.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/physics.hpp"
#include "simulation/simulation_manager.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace moonai {
namespace profiler {

struct WindowMeta {
  int index = 0;
  int predator_count = 0;
  int prey_count = 0;
  int species_count = 0;
  float best_fitness = 0.0f;
  float avg_fitness = 0.0f;
  float avg_complexity = 0.0f;
};

struct RunConfig {
  std::string experiment;
  std::string output_root;
  std::uint64_t seed = 0;
  int total_steps = 0;
  int report_interval = 0;
  bool gpu_allowed = false;
  bool cuda_compiled = false;
  bool openmp_compiled = false;
  std::string suite;
  std::string base_experiment;
  std::string config_fingerprint;
};

struct WindowRecord {
  WindowMeta meta;
  bool cpu_used = false;
  bool gpu_used = false;
  std::array<std::int64_t, event_count> durations_ns{};
};

class Profiler {
public:
  static Profiler &instance() {
    static Profiler profiler;
    return profiler;
  }

  void set_enabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_relaxed);
  }
  bool enabled() const {
    return enabled_.load(std::memory_order_relaxed);
  }

  void start_run(const RunConfig &cfg);
  void start_window(int window_index);
  void mark_cpu_used() {
    window_cpu_used_ = true;
  }
  void mark_gpu_used() {
    window_gpu_used_ = true;
  }
  void add_duration(ProfileEvent event, std::int64_t ns);
  void finish_window(const WindowMeta &meta);
  nlohmann::json finish_run(std::int64_t run_total_ns);

  const std::string &output_dir() const {
    return output_dir_;
  }

private:
  Profiler() = default;

  std::atomic<bool> enabled_{false};
  std::string experiment_;
  std::string output_dir_;
  std::string generated_at_utc_;
  std::uint64_t seed_ = 0;
  int total_steps_ = 0;
  int report_interval_ = 0;
  bool gpu_allowed_ = false;
  bool cuda_compiled_ = false;
  bool openmp_compiled_ = false;
  std::string suite_;
  std::string base_experiment_;
  std::string config_fingerprint_;

  bool window_cpu_used_ = false;
  bool window_gpu_used_ = false;
  bool window_active_ = false;
  std::chrono::steady_clock::time_point window_start_{};
  std::vector<WindowRecord> records_;
  std::array<std::atomic<std::int64_t>, event_count> current_durations_{};
};

// ScopedTimer implementation (declared in profiler_types.hpp under
// MOONAI_BUILD_PROFILER)
ScopedTimer::ScopedTimer(ProfileEvent event)
    : event_(event), active_(Profiler::instance().enabled()) {
  if (active_) {
    start_ = std::chrono::steady_clock::now();
  }
}

ScopedTimer::~ScopedTimer() {
  if (!active_)
    return;
  const auto end = std::chrono::steady_clock::now();
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
          .count();
  Profiler::instance().add_duration(event_, ns);
}

namespace detail {

std::string sanitize_path_component(const std::string &value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (char ch : value) {
    const bool valid = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                       (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
    sanitized.push_back(valid ? ch : '_');
  }
  return sanitized.empty() ? std::string{"profile"} : sanitized;
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

} // namespace detail

void Profiler::start_run(const RunConfig &cfg) {
  if (!enabled())
    return;

  const auto now = std::chrono::system_clock::now();
  experiment_ = cfg.experiment;
  generated_at_utc_ = detail::utc_timestamp_iso(now);
  seed_ = cfg.seed;
  total_steps_ = cfg.total_steps;
  report_interval_ = cfg.report_interval;
  gpu_allowed_ = cfg.gpu_allowed;
  cuda_compiled_ = cfg.cuda_compiled;
  openmp_compiled_ = cfg.openmp_compiled;
  suite_ = cfg.suite;
  base_experiment_ = cfg.base_experiment;
  config_fingerprint_ = cfg.config_fingerprint;
  window_cpu_used_ = false;
  window_gpu_used_ = false;
  window_active_ = false;
  records_.clear();

  try {
    const std::filesystem::path base_path(cfg.output_root);
    std::string run_name;
    if (!experiment_.empty()) {
      run_name = detail::utc_timestamp_for_path(now) + "_" +
                 detail::sanitize_path_component(experiment_);
    } else {
      run_name =
          detail::utc_timestamp_for_path(now) + "_seed" + std::to_string(seed_);
    }
    std::filesystem::path candidate = base_path / run_name;
    for (int suffix = 2; std::filesystem::exists(candidate); ++suffix) {
      candidate = base_path / (run_name + "_" + std::to_string(suffix));
    }
    std::filesystem::create_directories(candidate);
    output_dir_ = candidate.string();
  } catch (const std::exception &e) {
    output_dir_.clear();
    set_enabled(false);
    spdlog::error("Failed to initialize profiler output under '{}': {}",
                  cfg.output_root, e.what());
    return;
  }

  for (auto &d : current_durations_)
    d.store(0, std::memory_order_relaxed);
}

void Profiler::start_window(int window_index) {
  (void)window_index;
  if (!enabled())
    return;
  window_cpu_used_ = false;
  window_gpu_used_ = false;
  window_active_ = true;
  window_start_ = std::chrono::steady_clock::now();
  for (auto &d : current_durations_)
    d.store(0, std::memory_order_relaxed);
}

void Profiler::add_duration(ProfileEvent event, std::int64_t ns) {
  if (!enabled())
    return;
  current_durations_[static_cast<std::size_t>(event)].fetch_add(
      ns, std::memory_order_relaxed);
}

void Profiler::finish_window(const WindowMeta &meta) {
  if (!enabled())
    return;

  WindowRecord record;
  record.meta = meta;
  record.cpu_used = window_cpu_used_;
  record.gpu_used = window_gpu_used_;

  if (window_active_) {
    const auto window_end = std::chrono::steady_clock::now();
    const auto window_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               window_end - window_start_)
                               .count();
    current_durations_[static_cast<std::size_t>(ProfileEvent::WindowTotal)]
        .store(window_ns, std::memory_order_relaxed);
    window_active_ = false;
  }

  for (std::size_t i = 0; i < record.durations_ns.size(); ++i)
    record.durations_ns[i] =
        current_durations_[i].load(std::memory_order_relaxed);

  records_.push_back(std::move(record));
}

nlohmann::json Profiler::finish_run(std::int64_t run_total_ns) {
  if (!enabled())
    return nlohmann::json{};

  // Compute aggregates inline
  std::array<std::int64_t, event_count> total_durations{};
  std::array<int, event_count> active_duration_windows{};
  int cpu_windows = 0;
  int gpu_windows = 0;
  int min_predators = std::numeric_limits<int>::max();
  int max_predators = 0;
  int min_prey = std::numeric_limits<int>::max();
  int max_prey = 0;

  for (const auto &r : records_) {
    if (r.cpu_used)
      ++cpu_windows;
    if (r.gpu_used)
      ++gpu_windows;
    min_predators = std::min(min_predators, r.meta.predator_count);
    max_predators = std::max(max_predators, r.meta.predator_count);
    min_prey = std::min(min_prey, r.meta.prey_count);
    max_prey = std::max(max_prey, r.meta.prey_count);

    for (std::size_t i = 0; i < total_durations.size(); ++i) {
      total_durations[i] += r.durations_ns[i];
      if (r.durations_ns[i] > 0)
        ++active_duration_windows[i];
    }
  }

  // Build JSON output
  nlohmann::json profile;
  profile["schema_version"] = 4; // Removed counters
  profile["generated_at_utc"] = generated_at_utc_;
  profile["run"] = {{"experiment_name", experiment_},
                    {"base_experiment_name", base_experiment_},
                    {"suite_name", suite_},
                    {"output_dir", output_dir_},
                    {"seed", seed_},
                    {"config_fingerprint", config_fingerprint_},
                    {"total_steps", total_steps_},
                    {"report_interval_steps", report_interval_},
                    {"gpu_allowed", gpu_allowed_},
                    {"headless_only", true},
#ifdef _WIN32
                    {"platform", "windows"},
#else
                    {"platform", "linux"},
#endif
                    {"cuda_compiled", cuda_compiled_},
                    {"openmp_compiled", openmp_compiled_}};

  // Event definitions from constexpr array
  nlohmann::json event_defs = nlohmann::json::array();
  for (const auto &def : PROFILE_EVENTS) {
    event_defs.push_back({{"name", std::string(def.name)},
                          {"is_window_duration", def.is_window_duration},
                          {"description", std::string(def.description)}});
  }
  profile["event_definitions"] = std::move(event_defs);

  // Per-window data
  nlohmann::json window_rows = nlohmann::json::array();
  for (const auto &r : records_) {
    nlohmann::json durations;
    for (std::size_t i = 0; i < r.durations_ns.size(); ++i)
      durations[std::string(PROFILE_EVENTS[i].name)] =
          static_cast<double>(r.durations_ns[i]) / 1'000'000.0;

    window_rows.push_back({{"window_index", r.meta.index},
                           {"cpu_used", r.cpu_used},
                           {"gpu_used", r.gpu_used},
                           {"predator_count", r.meta.predator_count},
                           {"prey_count", r.meta.prey_count},
                           {"species_count", r.meta.species_count},
                           {"best_fitness", r.meta.best_fitness},
                           {"avg_fitness", r.meta.avg_fitness},
                           {"avg_complexity", r.meta.avg_complexity},
                           {"events_ms", std::move(durations)}});
  }
  profile["windows"] = std::move(window_rows);

  // Summary with pre-computed aggregates
  nlohmann::json summary;
  summary["window_count"] = static_cast<int>(records_.size());
  summary["run_total_ms"] = static_cast<double>(run_total_ns) / 1'000'000.0;
  summary["cpu_window_count"] = cpu_windows;
  summary["gpu_window_count"] = gpu_windows;
  summary["population_ranges"] = {
      {"predator_min", records_.empty() ? 0 : min_predators},
      {"predator_max", records_.empty() ? 0 : max_predators},
      {"prey_min", records_.empty() ? 0 : min_prey},
      {"prey_max", records_.empty() ? 0 : max_prey}};

  // Event aggregates
  nlohmann::json durations_json;
  for (std::size_t i = 0; i < total_durations.size(); ++i) {
    const double total_ms =
        static_cast<double>(total_durations[i]) / 1'000'000.0;
    const int active = active_duration_windows[i];
    durations_json[std::string(PROFILE_EVENTS[i].name)] = {
        {"total_ms", total_ms},
        {"avg_ms_per_window",
         records_.empty() ? 0.0
                          : total_ms / static_cast<double>(records_.size())},
        {"nonzero_window_count", active},
        {"avg_ms_per_nonzero_window",
         active == 0 ? 0.0 : total_ms / static_cast<double>(active)}};
  }
  summary["events"] = std::move(durations_json);
  profile["summary"] = std::move(summary);

  return profile;
}

void mark_cpu_used(bool used) {
  if (used)
    Profiler::instance().mark_cpu_used();
}
void mark_gpu_used(bool used) {
  if (used)
    Profiler::instance().mark_gpu_used();
}

struct SuiteConfig {
  std::string name;
  std::string config_path = "config.lua";
  std::string experiment;
  std::vector<std::uint64_t> seeds;
  int windows = 24;
  std::string output_dir = "output/profiles";
};

std::map<std::string, SuiteConfig>
load_suites_lua(const std::string &filepath) {
  std::map<std::string, SuiteConfig> suites;

  sol::state lua;
  lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::table,
                     sol::lib::string);

  try {
    sol::protected_function_result result = lua.safe_script_file(filepath);
    if (!result.valid()) {
      sol::error err = result;
      spdlog::error("Lua profiler config error in '{}': {}", filepath,
                    err.what());
      return suites;
    }

    sol::object obj = result;
    if (obj.get_type() != sol::type::table) {
      spdlog::error("Lua profiler config '{}' must return a table", filepath);
      return suites;
    }

    sol::table root = obj.as<sol::table>();
    for (auto &[key, value] : root) {
      if (key.get_type() != sol::type::string ||
          value.get_type() != sol::type::table) {
        continue;
      }

      SuiteConfig suite;
      suite.name = key.as<std::string>();
      const sol::table tbl = value.as<sol::table>();

      if (auto entry = tbl["config_path"]; entry.valid())
        suite.config_path = entry.get<std::string>();
      if (auto entry = tbl["experiment"]; entry.valid())
        suite.experiment = entry.get<std::string>();
      if (auto entry = tbl["windows"]; entry.valid())
        suite.windows = entry.get<int>();
      if (auto entry = tbl["output_dir"]; entry.valid())
        suite.output_dir = entry.get<std::string>();

      const sol::object seeds_obj = tbl["seeds"];
      if (seeds_obj.valid() && seeds_obj.get_type() == sol::type::table) {
        const sol::table seeds_tbl = seeds_obj.as<sol::table>();
        for (auto &[_, seed_value] : seeds_tbl) {
          if (!seed_value.valid())
            continue;
          suite.seeds.push_back(
              static_cast<std::uint64_t>(seed_value.as<double>()));
        }
      }

      if (suite.experiment.empty()) {
        spdlog::warn("Profiler suite '{}' missing required field 'experiment'",
                     suite.name);
        continue;
      }
      if (suite.seeds.empty()) {
        spdlog::warn("Profiler suite '{}' has no seeds", suite.name);
        continue;
      }

      suites[suite.name] = std::move(suite);
    }

    if (suites.empty()) {
      spdlog::error("Lua profiler config '{}' returned no named suites.",
                    filepath);
    } else {
      spdlog::info("Loaded {} profiler suite(s) from '{}'.", suites.size(),
                   filepath);
    }
  } catch (const std::exception &e) {
    spdlog::error("Failed to load Lua profiler config '{}': {}", filepath,
                  e.what());
  }

  return suites;
}

} // namespace profiler
} // namespace moonai

namespace {

struct Args {
  std::string profiler_config = "profiler.lua";
  std::string suite_name;
  bool no_gpu = false;
};

Args parse_args(int argc, const char *argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--suite" && i + 1 < argc) {
      args.suite_name = argv[++i];
    } else if (arg == "--no-gpu") {
      args.no_gpu = true;
    } else if (!arg.empty() && arg[0] != '-') {
      args.profiler_config = arg;
    }
  }
  return args;
}

struct RunResult {
  std::uint64_t seed = 0;
  double avg_window_ms = 0.0;
  double run_total_ms = 0.0;
  int window_count = 0;
  std::string disposition = "kept";
  nlohmann::json profile;
};

double json_number(const nlohmann::json &value) {
  return value.is_number() ? value.get<double>() : 0.0;
}

nlohmann::json aggregate_event_stats(const std::vector<const RunResult *> &runs,
                                     const std::string &key) {
  nlohmann::json aggregated = nlohmann::json::object();
  if (runs.empty())
    return aggregated;

  const auto &first = runs.front()->profile["summary"][key];
  if (!first.is_object())
    return aggregated;

  for (auto it = first.begin(); it != first.end(); ++it) {
    if (!it.value().is_object())
      continue;
    nlohmann::json stats = nlohmann::json::object();
    for (auto jt = it.value().begin(); jt != it.value().end(); ++jt) {
      double sum = 0.0;
      for (const auto *run : runs) {
        const auto &source = run->profile["summary"][key];
        if (source.contains(it.key()) && source[it.key()].contains(jt.key()))
          sum += json_number(source[it.key()][jt.key()]);
      }
      stats[jt.key()] = sum / static_cast<double>(runs.size());
    }
    aggregated[it.key()] = std::move(stats);
  }
  return aggregated;
}

std::string utc_timestamp_for_path() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc;
  gmtime_r(&time, &utc);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &utc);
  return std::string(buf);
}

RunResult run_suite_member(const moonai::profiler::SuiteConfig &suite,
                           moonai::SimulationConfig config, std::uint64_t seed,
                           bool no_gpu) {
  config.seed = seed;
  config.max_steps = suite.windows * config.report_interval_steps;

  moonai::Random rng(config.seed);
  moonai::SimulationManager simulation(config);
  moonai::EvolutionManager evolution(config, rng);
  moonai::MetricsCollector metrics;

  simulation.initialize();
  evolution.initialize(moonai::SensorInput::SIZE, 2);
  evolution.seed_initial_population(simulation);
  evolution.enable_gpu(!no_gpu);

  auto &profiler = moonai::profiler::Profiler::instance();
  profiler.set_enabled(true);

  // Initialize run config
  moonai::profiler::RunConfig run_cfg;
  run_cfg.experiment = suite.experiment;
  run_cfg.output_root = suite.output_dir;
  run_cfg.seed = seed;
  run_cfg.total_steps = config.max_steps;
  run_cfg.report_interval = config.report_interval_steps;
  run_cfg.gpu_allowed = !no_gpu;
#ifdef MOONAI_ENABLE_CUDA
  run_cfg.cuda_compiled = true;
#else
  run_cfg.cuda_compiled = false;
#endif
#ifdef MOONAI_OPENMP_ENABLED
  run_cfg.openmp_compiled = true;
#else
  run_cfg.openmp_compiled = false;
#endif
  run_cfg.suite = suite.name;
  run_cfg.base_experiment = suite.experiment;
  run_cfg.config_fingerprint = moonai::fingerprint_config(config);
  profiler.start_run(run_cfg);

  const auto run_start = std::chrono::steady_clock::now();
  const float dt = 1.0f / static_cast<float>(config.target_fps);
  std::vector<moonai::Vec2> actions;
  int steps_executed = 0;

  for (int window = 0; window < suite.windows; ++window) {
    profiler.start_window(window);
    const int window_end = std::min(
        config.max_steps, steps_executed + config.report_interval_steps);

    while (steps_executed < window_end) {
      // Try GPU full ecology first, fall back to CPU
      bool used_gpu = false;
      if (evolution.gpu_enabled()) {
        used_gpu = evolution.step_gpu(simulation, steps_executed);
      }

      if (!used_gpu) {
        // CPU path: compute actions and step simulation
        evolution.compute_actions(simulation, actions);
        for (std::size_t idx : simulation.alive_agent_indices())
          simulation.apply_action(idx, actions[idx], dt);
        simulation.step(dt);
      }

      const auto pairs = simulation.find_reproduction_pairs();
      for (const auto &pair : pairs)
        evolution.create_offspring(simulation, pair.parent_a, pair.parent_b,
                                   pair.spawn_position);

      evolution.refresh_fitness(simulation);
      ++steps_executed;
    }

    evolution.refresh_species(simulation);
    const auto snapshot = metrics.collect(steps_executed, simulation.agents(),
                                          0, 0, evolution.species_count());
    profiler.finish_window({window, snapshot.predator_count,
                            snapshot.prey_count, snapshot.num_species,
                            snapshot.best_fitness, snapshot.avg_fitness,
                            snapshot.avg_genome_complexity});
  }

  const auto run_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - run_start)
                          .count();
  auto profile_json = profiler.finish_run(run_ns);

  RunResult result;
  result.seed = seed;
  result.profile = std::move(profile_json);
  const auto &summary = result.profile["summary"];
  result.avg_window_ms =
      json_number(summary["events"]["window_total"]["avg_ms_per_window"]);
  result.run_total_ms = json_number(summary["run_total_ms"]);
  result.window_count = summary["window_count"].get<int>();
  return result;
}

void write_suite_manifest(const moonai::profiler::SuiteConfig &suite,
                          const moonai::SimulationConfig &config,
                          std::vector<RunResult> runs,
                          const std::filesystem::path &output_path) {
  std::sort(runs.begin(), runs.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.avg_window_ms < rhs.avg_window_ms;
  });

  std::vector<const RunResult *> kept;
  if (runs.size() > 2) {
    runs.front().disposition = "dropped_fastest";
    runs.back().disposition = "dropped_slowest";
    for (std::size_t i = 1; i + 1 < runs.size(); ++i)
      kept.push_back(&runs[i]);
  } else {
    for (auto &run : runs)
      kept.push_back(&run);
  }

  const double avg_window_ms =
      kept.empty() ? 0.0
                   : std::accumulate(kept.begin(), kept.end(), 0.0,
                                     [](double sum, const auto *r) {
                                       return sum + r->avg_window_ms;
                                     }) /
                         static_cast<double>(kept.size());
  const double avg_run_total_ms =
      kept.empty() ? 0.0
                   : std::accumulate(kept.begin(), kept.end(), 0.0,
                                     [](double sum, const auto *r) {
                                       return sum + r->run_total_ms;
                                     }) /
                         static_cast<double>(kept.size());

  double variance = 0.0;
  for (const auto *r : kept) {
    const double diff = r->avg_window_ms - avg_window_ms;
    variance += diff * diff;
  }
  const double stddev =
      kept.empty() ? 0.0
                   : std::sqrt(variance / static_cast<double>(kept.size()));

  nlohmann::json manifest;
  manifest["schema_version"] = 3;
  manifest["suite"] = {
      {"name", suite.name},
      {"config_path", suite.config_path},
      {"experiment_name", suite.experiment},
      {"windows", suite.windows},
      {"config_fingerprint", moonai::fingerprint_config(config)}};

  nlohmann::json run_rows = nlohmann::json::array();
  for (const auto &run : runs) {
    run_rows.push_back({{"seed", run.seed},
                        {"avg_window_ms", run.avg_window_ms},
                        {"run_total_ms", run.run_total_ms},
                        {"window_count", run.window_count},
                        {"disposition", run.disposition},
                        {"profile_data", run.profile}});
  }
  manifest["runs"] = std::move(run_rows);

  auto calc_avg = [&](const char *key) -> double {
    if (kept.empty())
      return 0.0;
    return std::accumulate(kept.begin(), kept.end(), 0.0,
                           [&](double sum, const auto *r) {
                             return sum +
                                    json_number(r->profile["summary"][key]);
                           }) /
           static_cast<double>(kept.size());
  };

  manifest["aggregate"] = {
      {"avg_window_ms", avg_window_ms},
      {"avg_window_ms_stddev", stddev},
      {"avg_run_total_ms", avg_run_total_ms},
      {"cpu_window_count_avg", calc_avg("cpu_window_count")},
      {"gpu_window_count_avg", calc_avg("gpu_window_count")},
      {"events", aggregate_event_stats(kept, "events")}};

  std::filesystem::create_directories(output_path.parent_path());
  std::ofstream file(output_path);
  if (!file.is_open()) {
    spdlog::error("Failed to open profiler output '{}'", output_path.string());
    return;
  }
  file << manifest.dump(2) << '\n';
  if (!file)
    spdlog::error("Failed to write profiler output '{}'", output_path.string());
}

} // namespace

int main(int argc, const char *argv[]) {
  const Args args = parse_args(argc, argv);
  if (args.suite_name.empty()) {
    std::fprintf(stderr,
                 "Usage: moonai_profiler profiler.lua --suite <name>\n");
    return 1;
  }

  auto suites = moonai::profiler::load_suites_lua(args.profiler_config);
  auto suite_it = suites.find(args.suite_name);
  if (suite_it == suites.end()) {
    spdlog::error("Profiler suite '{}' not found", args.suite_name);
    return 1;
  }

  const auto &suite = suite_it->second;
  auto configs = moonai::load_all_configs_lua(suite.config_path);
  auto config_it = configs.find(suite.experiment);
  if (config_it == configs.end()) {
    spdlog::error("Experiment '{}' not found in '{}'", suite.experiment,
                  suite.config_path);
    return 1;
  }

  std::vector<RunResult> runs;
  runs.reserve(suite.seeds.size());
  const auto output_path =
      std::filesystem::path(suite.output_dir) /
      (utc_timestamp_for_path() + "_" + suite.name + ".json");
  for (std::uint64_t seed : suite.seeds)
    runs.push_back(
        run_suite_member(suite, config_it->second, seed, args.no_gpu));

  write_suite_manifest(suite, config_it->second, std::move(runs), output_path);
  spdlog::info("Profiler output written to: {}", output_path.string());
  return 0;
}
