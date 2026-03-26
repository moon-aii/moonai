#include "core/config.hpp"
#include "core/profiler_macros.hpp"
#include "data/metrics.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/session.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
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
};

struct WindowRecord {
  WindowMeta meta;
  std::unordered_map<std::string, std::int64_t> durations_ns;
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
  void add_duration(const char *event_name, std::int64_t ns);
  void finish_window(const WindowMeta &meta);
  nlohmann::json finish_run(std::int64_t run_total_ns);

  int current_window() const {
    return static_cast<int>(records_.size());
  }

private:
  Profiler() = default;

  std::atomic<bool> enabled_{false};
  std::string experiment_;
  std::string generated_at_utc_;
  std::uint64_t seed_ = 0;
  int total_steps_ = 0;
  int report_interval_ = 0;
  bool gpu_allowed_ = false;
  bool cuda_compiled_ = false;
  bool openmp_compiled_ = false;
  std::string suite_;

  bool window_active_ = false;
  std::chrono::steady_clock::time_point window_start_{};
  std::vector<WindowRecord> records_;
  std::unordered_map<std::string, std::int64_t> current_durations_;
};

// ScopedTimer implementation (declared in profiler_macros.hpp)
ScopedTimer::ScopedTimer(const char *event_name)
    : event_name_(event_name), active_(Profiler::instance().enabled()) {
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
  Profiler::instance().add_duration(event_name_, ns);
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
  oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
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
  window_active_ = false;
  records_.clear();
  current_durations_.clear();
}

void Profiler::start_window(int window_index) {
  (void)window_index;
  if (!enabled())
    return;
  window_active_ = true;
  window_start_ = std::chrono::steady_clock::now();
  current_durations_.clear();
}

void Profiler::add_duration(const char *event_name, std::int64_t ns) {
  if (!enabled())
    return;
  current_durations_[event_name] += ns;
}

void Profiler::finish_window(const WindowMeta &meta) {
  if (!enabled())
    return;

  WindowRecord record;
  record.meta = meta;

  if (window_active_) {
    const auto window_end = std::chrono::steady_clock::now();
    const auto window_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               window_end - window_start_)
                               .count();
    record.durations_ns["window_total"] = window_ns;
    window_active_ = false;
  }

  for (const auto &[name, value] : current_durations_) {
    record.durations_ns[name] = value;
  }

  records_.push_back(std::move(record));
}

nlohmann::json Profiler::finish_run(std::int64_t run_total_ns) {
  if (!enabled())
    return nlohmann::json{};

  // Build JSON output - raw data only, no aggregation
  nlohmann::json profile;
  profile["schema_version"] = 1;
  profile["generated_at_utc"] = generated_at_utc_;
  profile["run"] = {{"experiment_name", experiment_},
                    {"suite_name", suite_},
                    {"seed", seed_},
                    {"total_steps", total_steps_},
                    {"report_interval_steps", report_interval_},
                    {"gpu_allowed", gpu_allowed_},
                    {"headless_only", false}, // Now runs with GUI
#ifdef _WIN32
                    {"platform", "windows"},
#else
                    {"platform", "linux"},
#endif
                    {"cuda_compiled", cuda_compiled_},
                    {"openmp_compiled", openmp_compiled_}};

  // Per-window raw data
  nlohmann::json window_rows = nlohmann::json::array();
  for (const auto &r : records_) {
    nlohmann::json durations;
    for (const auto &[name, ns] : r.durations_ns) {
      durations[name] = static_cast<double>(ns) / 1'000'000.0;
    }

    window_rows.push_back({{"window_index", r.meta.index},
                           {"predator_count", r.meta.predator_count},
                           {"prey_count", r.meta.prey_count},
                           {"species_count", r.meta.species_count},
                           {"best_fitness", r.meta.best_fitness},
                           {"avg_fitness", r.meta.avg_fitness},
                           {"avg_complexity", r.meta.avg_complexity},
                           {"events_ms", std::move(durations)}});
  }
  profile["windows"] = std::move(window_rows);

  // Minimal summary - just timing info
  nlohmann::json summary;
  summary["window_count"] = static_cast<int>(records_.size());
  summary["run_total_ms"] = static_cast<double>(run_total_ns) / 1'000'000.0;
  profile["summary"] = std::move(summary);

  return profile;
}

} // namespace profiler
} // namespace moonai

namespace {

struct Args {
  int windows = 24;
  std::vector<std::uint64_t> seeds = {41, 42, 43, 44, 45, 46};
  std::string output_dir = "output/profiles";
  std::string experiment_name = "profile";
  bool no_gpu = false;
  bool headless = false; // GUI mode by default
};

std::vector<std::uint64_t> parse_seeds(const std::string &str) {
  std::vector<std::uint64_t> seeds;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    // Trim whitespace
    item.erase(0, item.find_first_not_of(" \t"));
    item.erase(item.find_last_not_of(" \t") + 1);
    if (!item.empty()) {
      seeds.push_back(static_cast<std::uint64_t>(std::stoull(item)));
    }
  }
  return seeds;
}

Args parse_args(int argc, const char *argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--windows" && i + 1 < argc) {
      args.windows = std::stoi(argv[++i]);
    } else if (arg == "--seeds" && i + 1 < argc) {
      args.seeds = parse_seeds(argv[++i]);
    } else if (arg == "--output-dir" && i + 1 < argc) {
      args.output_dir = argv[++i];
    } else if (arg == "--name" && i + 1 < argc) {
      args.experiment_name = argv[++i];
    } else if (arg == "--no-gpu") {
      args.no_gpu = true;
    } else if (arg == "--headless") {
      args.headless = true;
    }
  }
  return args;
}

struct RunResult {
  std::uint64_t seed = 0;
  nlohmann::json profile;
};

std::string utc_timestamp_for_path() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc;
  gmtime_r(&time, &utc);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &utc);
  return std::string(buf);
}

RunResult run_profiler(const std::string &experiment_name,
                       const std::string &output_dir, int windows,
                       std::uint64_t seed, bool no_gpu, bool headless) {
  using namespace moonai;

  auto &profiler = profiler::Profiler::instance();
  profiler.set_enabled(true);

  // Window tracking state
  int current_window = 0;

  // Build SessionConfig for profiling with GUI
  SessionConfig session_cfg;
  session_cfg.sim_config = SimulationConfig(); // default config
  session_cfg.sim_config.seed = seed;
  session_cfg.sim_config.max_steps =
      windows * session_cfg.sim_config.report_interval_steps;
  session_cfg.experiment_name = experiment_name;
  session_cfg.output_dir = output_dir;
  session_cfg.seed = seed;
  session_cfg.headless = headless;
  session_cfg.enable_gpu = !no_gpu;
  session_cfg.enable_interactions = false; // Disable pause, step, selection
  session_cfg.auto_run = true;             // Run continuously
  session_cfg.speed_multiplier = 1;        // Normal speed

  // Set up profiler window tracking via callback
  session_cfg.on_report_callback = [&profiler, &current_window,
                                    windows](const StepMetrics &snapshot) {
    if (current_window < windows) {
      profiler.finish_window({current_window, snapshot.predator_count,
                              snapshot.prey_count, snapshot.num_species,
                              snapshot.best_fitness, snapshot.avg_fitness,
                              snapshot.avg_genome_complexity});
      current_window++;
      if (current_window < windows) {
        profiler.start_window(current_window);
      }
    }
  };

  // Check for display if not headless
  if (!headless && std::getenv("DISPLAY") == nullptr &&
      std::getenv("WAYLAND_DISPLAY") == nullptr) {
    spdlog::warn("No display server found; switching to headless mode.");
    session_cfg.headless = true;
  }

  // Initialize profiler run config
  profiler::RunConfig run_cfg;
  run_cfg.experiment = experiment_name;
  run_cfg.output_root = output_dir;
  run_cfg.seed = seed;
  run_cfg.total_steps = session_cfg.sim_config.max_steps;
  run_cfg.report_interval = session_cfg.sim_config.report_interval_steps;
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
  run_cfg.suite = experiment_name;
  profiler.start_run(run_cfg);

  const auto run_start = std::chrono::steady_clock::now();

  // Start first window
  profiler.start_window(0);

  // Create Session and run - signals handled internally
  Session session(session_cfg);
  auto stop_reason = session.run();

  // If stopped early, fill remaining windows with empty data
  while (current_window < windows) {
    // Get current state for the remaining windows
    auto &evolution = session.evolution();
    auto &metrics = session.metrics();
    auto &registry = session.registry();

    evolution.refresh_species_ecs(registry);
    const auto snapshot =
        metrics.collect_ecs(session.steps_executed(), registry, evolution, 0, 0,
                            evolution.species_count());
    profiler.finish_window({current_window, snapshot.predator_count,
                            snapshot.prey_count, snapshot.num_species,
                            snapshot.best_fitness, snapshot.avg_fitness,
                            snapshot.avg_genome_complexity});
    current_window++;
    if (current_window < windows) {
      profiler.start_window(current_window);
    }
  }

  const auto run_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - run_start)
                          .count();
  auto profile_json = profiler.finish_run(run_ns);

  RunResult result;
  result.seed = seed;
  result.profile = std::move(profile_json);

  if (stop_reason == StopReason::UserQuit) {
    spdlog::info("Run stopped by user (window closed)");
  } else if (stop_reason == StopReason::Signal) {
    spdlog::info("Run stopped by signal (Ctrl+C)");
  }

  return result;
}

void write_manifest(const std::string &experiment_name, int windows,
                    std::vector<RunResult> runs,
                    const std::filesystem::path &output_path) {
  nlohmann::json manifest;
  manifest["schema_version"] = 1;
  manifest["suite"] = {{"name", experiment_name}, {"windows", windows}};

  nlohmann::json run_rows = nlohmann::json::array();
  for (const auto &run : runs) {
    run_rows.push_back({{"seed", run.seed}, {"profile_data", run.profile}});
  }
  manifest["runs"] = std::move(run_rows);

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

  std::vector<RunResult> runs;
  runs.reserve(args.seeds.size());
  const auto output_path =
      std::filesystem::path(args.output_dir) /
      (utc_timestamp_for_path() + "_" + args.experiment_name + ".json");

  for (std::uint64_t seed : args.seeds) {
    runs.push_back(run_profiler(args.experiment_name, args.output_dir,
                                args.windows, seed, args.no_gpu,
                                args.headless));
  }

  write_manifest(args.experiment_name, args.windows, std::move(runs),
                 output_path);
  spdlog::info("Profiler output written to: {}", output_path.string());
  return 0;
}

// Profiler macros - defined at end of file
#ifndef MOONAI_BUILD_PROFILER

#define MOONAI_PROFILE_SCOPE(event_name) ((void)0)

#else

#define MOONAI_PROFILE_SCOPE(event_name)                                       \
  ::moonai::profiler::ScopedTimer _moonai_scoped_timer(event_name)

#endif
