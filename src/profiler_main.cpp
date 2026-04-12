#include "app/app.hpp"
#include "core/config.hpp"
#include "core/profiler_macros.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// CUDA includes for GPU profiling
#ifdef MOONAI_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc;
#ifdef _WIN32
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &utc);
  return std::string(buf);
}

// Constants for time conversion
constexpr double NS_TO_MS = 1'000'000.0;
inline double ns_to_ms(std::int64_t ns) {
  return static_cast<double>(ns) / NS_TO_MS;
}

namespace moonai {
namespace profiler {

struct ScopeNode {
  std::string name;
  std::chrono::steady_clock::time_point start;
  std::int64_t duration_ns = 0;
  std::vector<std::unique_ptr<ScopeNode>> children;
  ScopeNode *parent = nullptr;
};

#ifdef MOONAI_ENABLE_CUDA
struct GpuEventPair {
  const char *name;
  cudaEvent_t start;
  cudaEvent_t end;
  cudaStream_t stream;
};
#endif

class Profiler {
public:
  static Profiler &instance() {
    static Profiler profiler;
    return profiler;
  }

  void start_run(const AppConfig &cfg);
  void begin_scope(const char *event_name);
  void end_scope(const char *event_name);

#ifdef MOONAI_ENABLE_CUDA
  int record_gpu_event_start(const char *name, cudaStream_t stream);
  void record_gpu_event_end(int event_index);
  void merge_gpu_timings();
#endif

  std::vector<std::unique_ptr<ScopeNode>> finish_run();

private:
  Profiler();
  ~Profiler();

  std::optional<AppConfig> cfg_;
  std::vector<std::unique_ptr<ScopeNode>> frame_trees_;
  std::vector<ScopeNode *> active_stack_;
  std::unique_ptr<ScopeNode> pending_root_;

#ifdef MOONAI_ENABLE_CUDA
  std::vector<GpuEventPair> pending_gpu_events_;
  int next_gpu_event_index_;
#endif
};

Profiler::Profiler()
#ifdef MOONAI_ENABLE_CUDA
    : next_gpu_event_index_(0)
#endif
{
}

Profiler::~Profiler() {
#ifdef MOONAI_ENABLE_CUDA
  for (auto &event : pending_gpu_events_) {
    if (event.start)
      cudaEventDestroy(event.start);
    if (event.end)
      cudaEventDestroy(event.end);
  }
#endif
}

ScopedTimer::ScopedTimer(const char *event_name, cudaStream_t stream)
    : event_name_(event_name), stream_(stream), gpu_event_index_(-1), has_cpu_scope_(stream == nullptr) {
#ifdef MOONAI_ENABLE_CUDA
  if (stream) {
    gpu_event_index_ = Profiler::instance().record_gpu_event_start(event_name, stream);
  }
#endif
  if (has_cpu_scope_) {
    Profiler::instance().begin_scope(event_name);
  }
}

ScopedTimer::~ScopedTimer() {
  if (std::strcmp(event_name_, "gpu_synchronize") == 0) {
#ifdef MOONAI_ENABLE_CUDA
    Profiler::instance().merge_gpu_timings();
#endif
  }

#ifdef MOONAI_ENABLE_CUDA
  if (gpu_event_index_ >= 0) {
    Profiler::instance().record_gpu_event_end(gpu_event_index_);
  }
#endif

  if (has_cpu_scope_) {
    Profiler::instance().end_scope(event_name_);
  }
}

void Profiler::start_run(const AppConfig &cfg) {
  cfg_ = cfg;
  frame_trees_.clear();
  active_stack_.clear();
  pending_root_.reset();
#ifdef MOONAI_ENABLE_CUDA
  pending_gpu_events_.clear();
  next_gpu_event_index_ = 0;
#endif
}

void Profiler::begin_scope(const char *event_name) {
  auto node = std::make_unique<ScopeNode>();
  node->name = event_name;
  node->start = std::chrono::steady_clock::now();

  if (active_stack_.empty()) {
    node->parent = nullptr;
    pending_root_ = std::move(node);
    active_stack_.push_back(pending_root_.get());
  } else {
    node->parent = active_stack_.back();
    ScopeNode *node_ptr = node.get();
    active_stack_.back()->children.push_back(std::move(node));
    active_stack_.push_back(node_ptr);
  }
}

void Profiler::end_scope(const char *event_name) {
  assert(!active_stack_.empty() && "Scope stack underflow: no active scope to end");
  assert(active_stack_.back()->name == event_name && "Scope mismatch: expected scope to match the ending scope name");

  ScopeNode *node = active_stack_.back();
  active_stack_.pop_back();

  const auto end = std::chrono::steady_clock::now();
  node->duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - node->start).count();

  if (std::strcmp(event_name, "frame_total") == 0) {
    assert(active_stack_.empty() && "frame_total ended but scope stack not empty");
    assert(pending_root_ != nullptr && "Root node should exist");
    frame_trees_.push_back(std::move(pending_root_));
    pending_root_.reset();
  }
}

#ifdef MOONAI_ENABLE_CUDA
int Profiler::record_gpu_event_start(const char *name, cudaStream_t stream) {
  int index = next_gpu_event_index_++;

  if (index >= static_cast<int>(pending_gpu_events_.size())) {
    pending_gpu_events_.resize(index + 1);
  }

  GpuEventPair &pair = pending_gpu_events_[index];
  pair.name = name;
  pair.stream = stream;

  cudaEventCreate(&pair.start);
  cudaEventRecord(pair.start, stream);
  pair.end = nullptr;

  return index;
}

void Profiler::record_gpu_event_end(int event_index) {
  assert(event_index >= 0 && event_index < static_cast<int>(pending_gpu_events_.size()));

  GpuEventPair &pair = pending_gpu_events_[event_index];
  cudaEventCreate(&pair.end);
  cudaEventRecord(pair.end, pair.stream);
}

void Profiler::merge_gpu_timings() {
  if (pending_gpu_events_.empty()) {
    return;
  }

  for (const auto &pair : pending_gpu_events_) {
    if (!pair.start || !pair.end) {
      continue;
    }

    float elapsed_ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&elapsed_ms, pair.start, pair.end);

    if (err == cudaSuccess && elapsed_ms > 0.0f) {
      auto gpu_node = std::make_unique<ScopeNode>();
      gpu_node->name = std::string(pair.name) + "_gpu";
      gpu_node->duration_ns = static_cast<std::int64_t>(elapsed_ms * NS_TO_MS);
      gpu_node->parent = active_stack_.back();
      active_stack_.back()->children.push_back(std::move(gpu_node));
    }

    cudaEventDestroy(pair.start);
    cudaEventDestroy(pair.end);
  }

  pending_gpu_events_.clear();
  next_gpu_event_index_ = 0;
}
#endif

std::vector<std::unique_ptr<ScopeNode>> Profiler::finish_run() {
  return std::move(frame_trees_);
}

} // namespace profiler
} // namespace moonai

namespace {

struct ProfilerArgs {
  int frames = 600;
  int speed_multiplier = 64;
  std::vector<std::uint64_t> seeds = {61, 62, 63, 64, 65, 66};
  std::string output_dir = "output/profiles";
  std::string experiment_name = "profile";
  bool no_gpu = false;
  bool help = false;
};

struct ParseProfilerResult {
  ProfilerArgs args;
  bool ok = true;
};

void print_profiler_usage(const char *program_name) {
  std::printf("MoonAI profiler\n"
              "\n"
              "Usage: %s [OPTIONS]\n"
              "\n"
              "Options:\n"
              "      --frames <n>          Number of frames per run (default: 600)\n"
              "      --output-dir <path>   Output directory for profiler JSON\n"
              "      --name <name>         Suite name in output file\n"
              "      --no-gpu              Disable CUDA GPU acceleration\n"
              "  -h, --help                Show this help message\n",
              program_name);
}

bool parse_int_arg(const char *value, int &out) {
  try {
    out = std::stoi(value);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

ParseProfilerResult parse_profiler_args(int argc, const char *argv[]) {
  ParseProfilerResult result;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      result.args.help = true;
      continue;
    }

    if (arg == "--frames") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
        result.ok = false;
        continue;
      }

      int parsed = 0;
      if (!parse_int_arg(argv[++i], parsed)) {
        std::fprintf(stderr, "Invalid frames value '%s'\n", argv[i]);
        result.ok = false;
        continue;
      }
      result.args.frames = parsed;
    } else if (arg == "--output-dir") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
        result.ok = false;
        continue;
      }
      result.args.output_dir = argv[++i];
    } else if (arg == "--name") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
        result.ok = false;
        continue;
      }
      result.args.experiment_name = argv[++i];
    } else if (arg == "--no-gpu") {
      result.args.no_gpu = true;
    } else {
      spdlog::warn("Unknown profiler argument: {}", arg);
    }
  }

  return result;
}

struct RunData {
  std::uint64_t seed = 0;
  std::vector<std::unique_ptr<moonai::profiler::ScopeNode>> frames;
  double avg_frame_ms = 0.0;
  double run_total_ms = 0.0;
  std::string disposition;
};

struct AveragedNode {
  std::string name;
  double avg_ms = 0.0;
  double total_ms = 0.0;
  int count = 0;
  double pct_of_parent = 0.0;
  std::vector<std::unique_ptr<AveragedNode>> children;
};

RunData run_profiler(const moonai::AppConfig &cfg) {
  using namespace moonai;

  auto &profiler = profiler::Profiler::instance();
  profiler.start_run(cfg);

  const auto run_start = std::chrono::steady_clock::now();

  App app(cfg);
  bool completed = app.run();

  if (!completed) {
    spdlog::info("Run stopped early, skipping profile generation");
    RunData skipped;
    skipped.seed = static_cast<std::uint64_t>(cfg.sim_config.seed);
    return skipped;
  }

  RunData result;
  result.seed = cfg.sim_config.seed;

  const auto run_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - run_start).count();
  result.run_total_ms = ns_to_ms(run_ns);
  result.frames = profiler.finish_run();

  // Calculate average frame time
  if (!result.frames.empty()) {
    double total = 0.0;
    for (const auto &frame : result.frames) {
      total += ns_to_ms(frame->duration_ns);
    }
    result.avg_frame_ms = total / result.frames.size();
  }

  return result;
}

nlohmann::json serialize_averaged_node(const AveragedNode *node) {
  nlohmann::json j;
  j["name"] = node->name;
  j["avg_ms"] = node->avg_ms;
  j["total_ms"] = node->total_ms;
  j["count"] = node->count;
  j["pct_of_parent"] = node->pct_of_parent;

  nlohmann::json children = nlohmann::json::array();
  for (const auto &child : node->children) {
    children.push_back(serialize_averaged_node(child.get()));
  }
  j["children"] = std::move(children);

  return j;
}

void collect_scope_nodes(const moonai::profiler::ScopeNode *node,
                         std::vector<const moonai::profiler::ScopeNode *> &out) {
  out.push_back(node);
  for (const auto &child : node->children) {
    collect_scope_nodes(child.get(), out);
  }
}

std::unique_ptr<AveragedNode> merge_nodes(const std::vector<const moonai::profiler::ScopeNode *> &nodes) {
  if (nodes.empty())
    return nullptr;

  auto result = std::make_unique<AveragedNode>();
  result->name = nodes[0]->name;

  double total = 0.0;
  for (const auto *node : nodes) {
    total += ns_to_ms(node->duration_ns);
  }

  result->total_ms = total;
  result->count = static_cast<int>(nodes.size());
  result->avg_ms = total / nodes.size();

  // Group children by name
  std::unordered_map<std::string, std::vector<const moonai::profiler::ScopeNode *>> child_groups;
  for (const auto *node : nodes) {
    for (const auto &child : node->children) {
      child_groups[child->name].push_back(child.get());
    }
  }

  // Recursively merge children
  for (auto &[name, group] : child_groups) {
    result->children.push_back(merge_nodes(group));
  }

  return result;
}

void calculate_percentages(AveragedNode *node, double parent_total) {
  if (parent_total > 0.0) {
    node->pct_of_parent = (node->total_ms / parent_total) * 100.0;
  } else {
    node->pct_of_parent = 100.0; // Root node
  }

  for (auto &child : node->children) {
    calculate_percentages(child.get(), node->total_ms);
  }
}

std::unique_ptr<AveragedNode> build_averaged_tree(const std::vector<RunData> &kept_runs) {
  if (kept_runs.empty() || kept_runs[0].frames.empty()) {
    return nullptr;
  }

  // Collect all frame roots from all kept runs
  std::vector<const moonai::profiler::ScopeNode *> all_roots;
  for (const auto &run : kept_runs) {
    for (const auto &frame : run.frames) {
      all_roots.push_back(frame.get());
    }
  }

  auto tree = merge_nodes(all_roots);
  if (tree) {
    calculate_percentages(tree.get(), 0.0);
  }

  return tree;
}

std::vector<double> build_frame_timeline(const std::vector<RunData> &kept_runs) {
  if (kept_runs.empty() || kept_runs[0].frames.empty()) {
    return {};
  }

  size_t frame_count = kept_runs[0].frames.size();
  std::vector<double> timeline(frame_count, 0.0);

  for (size_t i = 0; i < frame_count; ++i) {
    double sum = 0.0;
    for (const auto &run : kept_runs) {
      if (i < run.frames.size()) {
        sum += ns_to_ms(run.frames[i]->duration_ns);
      }
    }
    timeline[i] = sum / kept_runs.size();
  }

  return timeline;
}

void write_manifest(std::vector<RunData> runs, const std::filesystem::path &output_path, const moonai::AppConfig &cfg) {
  nlohmann::json manifest;

  // Metadata
  nlohmann::json metadata;
  metadata["suite_name"] = cfg.experiment_name;
  metadata["frame_count"] = cfg.sim_config.max_steps / cfg.speed_multiplier;
  metadata["gpu_allowed"] = cfg.enable_gpu;
  metadata["platform"] = moonai::AppConfig::platform;
  metadata["cuda_compiled"] = moonai::AppConfig::cuda_compiled;
  metadata["generated_at_utc"] = utc_timestamp();
  manifest["metadata"] = metadata;

  // Filter runs that have frames (completed successfully) and sort by
  // avg_frame_ms
  std::vector<RunData> completed;
  for (auto &run : runs) {
    if (!run.frames.empty()) {
      completed.push_back(std::move(run));
    }
  }

  if (completed.empty()) {
    spdlog::error("No completed runs to write");
    return;
  }

  std::sort(completed.begin(), completed.end(),
            [](const RunData &a, const RunData &b) { return a.avg_frame_ms < b.avg_frame_ms; });

  // Mark dispositions
  if (completed.size() > 2) {
    completed[0].disposition = "dropped_fastest";
    completed.back().disposition = "dropped_slowest";
    for (size_t i = 1; i < completed.size() - 1; ++i) {
      completed[i].disposition = "kept";
    }
  } else {
    for (auto &run : completed) {
      run.disposition = "kept";
    }
  }

  // Build runs list with dispositions (do this before moving kept runs)
  nlohmann::json run_array = nlohmann::json::array();
  for (const auto &run : completed) {
    nlohmann::json run_obj;
    run_obj["seed"] = run.seed;
    run_obj["avg_frame_ms"] = run.avg_frame_ms;
    run_obj["disposition"] = run.disposition;
    run_array.push_back(run_obj);
  }
  manifest["runs"] = run_array;

  // Collect kept runs (move frames from completed to kept)
  std::vector<RunData> kept;
  for (auto &run : completed) {
    if (run.disposition == "kept") {
      kept.push_back(std::move(run));
    }
  }

  // Summary statistics from kept runs
  double sum_avg = 0.0;
  double min_frame = kept.empty() ? 0.0 : kept[0].avg_frame_ms;
  double max_frame = kept.empty() ? 0.0 : kept[0].avg_frame_ms;
  for (const auto &run : kept) {
    sum_avg += run.avg_frame_ms;
    min_frame = std::min(min_frame, run.avg_frame_ms);
    max_frame = std::max(max_frame, run.avg_frame_ms);
  }
  double avg_frame_ms = kept.empty() ? 0.0 : sum_avg / kept.size();

  // Standard deviation
  double variance = 0.0;
  for (const auto &run : kept) {
    variance += std::pow(run.avg_frame_ms - avg_frame_ms, 2);
  }
  double stddev = kept.empty() ? 0.0 : std::sqrt(variance / kept.size());

  nlohmann::json summary;
  summary["avg_frame_ms"] = avg_frame_ms;
  summary["stddev_ms"] = stddev;
  summary["min_frame_ms"] = min_frame;
  summary["max_frame_ms"] = max_frame;
  manifest["summary"] = summary;

  // Build and serialize averaged tree
  auto tree = build_averaged_tree(kept);
  if (tree) {
    manifest["tree"] = serialize_averaged_node(tree.get());
  }

  // Build frame timeline
  auto timeline = build_frame_timeline(kept);
  manifest["frame_timeline_ms"] = timeline;

  std::filesystem::create_directories(output_path.parent_path());
  std::ofstream file(output_path);
  if (!file.is_open()) {
    spdlog::error("Failed to open profiler output '{}'", output_path.string());
    return;
  }
  file << manifest.dump(2) << '\n';
  if (!file)
    spdlog::error("Failed to write profiler output '{}'", output_path.string());
  else
    spdlog::info("Profiler output written to: {}", output_path.string());
}

} // namespace

int main(int argc, const char *argv[]) {
  const auto parsed = parse_profiler_args(argc, argv);
  const auto &args = parsed.args;

  if (args.help) {
    print_profiler_usage(argv[0]);
    return parsed.ok ? 0 : 1;
  }

  if (!parsed.ok) {
    print_profiler_usage(argv[0]);
    return 1;
  }

  moonai::AppConfig base_cfg;
  base_cfg.sim_config = moonai::SimulationConfig();
  base_cfg.sim_config.max_steps = args.frames * args.speed_multiplier;
  base_cfg.experiment_name = args.experiment_name;
  base_cfg.headless = false;
  base_cfg.enable_gpu = !args.no_gpu;
  base_cfg.speed_multiplier = args.speed_multiplier;
  const auto output_path =
      std::filesystem::path(args.output_dir) / (utc_timestamp() + "_" + args.experiment_name + ".json");

  std::vector<RunData> runs;
  runs.reserve(args.seeds.size());
  for (std::uint64_t seed : args.seeds) {
    base_cfg.sim_config.seed = seed;
    runs.push_back(run_profiler(base_cfg));
  }

  write_manifest(std::move(runs), output_path, base_cfg);
  return 0;
}
