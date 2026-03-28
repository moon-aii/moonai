#include "app.hpp"
#include "core/config.hpp"
#include "core/profiler_macros.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
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

namespace moonai {
namespace profiler {

struct ScopeNode {
  std::string name;
  std::chrono::steady_clock::time_point start;
  std::int64_t inclusive_ns = 0;
  std::int64_t exclusive_ns = 0;
  std::vector<std::unique_ptr<ScopeNode>> children;
  ScopeNode *parent = nullptr;
};

#ifdef MOONAI_ENABLE_CUDA
// GPU event tracking structure
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
  // GPU event tracking
  int record_gpu_event_start(const char *name, cudaStream_t stream);
  void record_gpu_event_end(int event_index);
  void merge_gpu_timings(); // Calculate GPU times and add as children
#endif

  nlohmann::json finish_run(std::int64_t run_total_ns);

private:
  Profiler();
  ~Profiler();

  std::optional<AppConfig> cfg_;
  std::vector<std::unique_ptr<ScopeNode>> frame_trees_;
  std::vector<ScopeNode *> active_stack_;
  std::unique_ptr<ScopeNode> pending_root_;

#ifdef MOONAI_ENABLE_CUDA
  // GPU event storage
  std::vector<GpuEventPair> pending_gpu_events_;
  int next_gpu_event_index_;
#endif

  nlohmann::json serialize_node(const ScopeNode *node) const;
};

// Profiler constructor/destructor
Profiler::Profiler()
#ifdef MOONAI_ENABLE_CUDA
    : next_gpu_event_index_(0)
#endif
{
}

Profiler::~Profiler() {
#ifdef MOONAI_ENABLE_CUDA
  // Clean up any remaining GPU events
  for (auto &event : pending_gpu_events_) {
    if (event.start)
      cudaEventDestroy(event.start);
    if (event.end)
      cudaEventDestroy(event.end);
  }
#endif
}

// ScopedTimer implementation (declared in profiler_macros.hpp)
ScopedTimer::ScopedTimer(const char *event_name, cudaStream_t stream)
    : event_name_(event_name), stream_(stream), gpu_event_index_(-1),
      has_cpu_scope_(stream == nullptr) // No CPU scope when stream provided
{
#ifdef MOONAI_ENABLE_CUDA
  if (stream) {
    gpu_event_index_ =
        Profiler::instance().record_gpu_event_start(event_name, stream);
  }
#endif
  if (has_cpu_scope_) {
    Profiler::instance().begin_scope(event_name);
  }
}

ScopedTimer::~ScopedTimer() {
  // Check if this is gpu_synchronize - merge GPU timings before ending scope
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
    // This is a root scope
    node->parent = nullptr;
    pending_root_ = std::move(node);
    active_stack_.push_back(pending_root_.get());
  } else {
    // This is a child scope
    node->parent = active_stack_.back();
    ScopeNode *node_ptr = node.get();
    active_stack_.back()->children.push_back(std::move(node));
    active_stack_.push_back(node_ptr);
  }
}

void Profiler::end_scope(const char *event_name) {
  assert(!active_stack_.empty() &&
         "Scope stack underflow: no active scope to end");
  assert(active_stack_.back()->name == event_name &&
         "Scope mismatch: expected scope to match the ending scope name");

  ScopeNode *node = active_stack_.back();
  active_stack_.pop_back();

  const auto end = std::chrono::steady_clock::now();
  node->inclusive_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - node->start)
          .count();

  // Calculate exclusive time: inclusive - sum(children inclusive)
  std::int64_t children_sum = 0;
  for (const auto &child : node->children) {
    children_sum += child->inclusive_ns;
  }
  node->exclusive_ns = node->inclusive_ns - children_sum;

  // If this is frame_total (root), commit the frame
  if (std::strcmp(event_name, "frame_total") == 0) {
    assert(active_stack_.empty() &&
           "frame_total ended but scope stack not empty - scopes not properly "
           "nested");
    assert(pending_root_ != nullptr && "Root node should exist");

    // Move the root node to frame_trees
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
  pair.end = nullptr; // Will be created on end

  return index;
}

void Profiler::record_gpu_event_end(int event_index) {
  assert(event_index >= 0 &&
         event_index < static_cast<int>(pending_gpu_events_.size()));

  GpuEventPair &pair = pending_gpu_events_[event_index];
  cudaEventCreate(&pair.end);
  cudaEventRecord(pair.end, pair.stream);
}

void Profiler::merge_gpu_timings() {
  if (pending_gpu_events_.empty()) {
    return;
  }

  // All GPU events should be completed by now (synchronize was called)
  for (const auto &pair : pending_gpu_events_) {
    if (!pair.start || !pair.end) {
      continue; // Incomplete event
    }

    float elapsed_ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&elapsed_ms, pair.start, pair.end);

    if (err == cudaSuccess && elapsed_ms > 0.0f) {
      // Create a child node for this GPU timing
      auto gpu_node = std::make_unique<ScopeNode>();
      gpu_node->name = std::string(pair.name) + "_gpu";
      gpu_node->inclusive_ns =
          static_cast<std::int64_t>(elapsed_ms * 1'000'000.0f);
      gpu_node->exclusive_ns = gpu_node->inclusive_ns; // No children yet
      gpu_node->parent = active_stack_.back();

      // Add as child to current scope
      active_stack_.back()->children.push_back(std::move(gpu_node));
    }

    // Clean up CUDA events
    cudaEventDestroy(pair.start);
    cudaEventDestroy(pair.end);
  }

  // Clear for next frame
  pending_gpu_events_.clear();
  next_gpu_event_index_ = 0;
}
#endif

nlohmann::json Profiler::serialize_node(const ScopeNode *node) const {
  nlohmann::json j;
  j["name"] = node->name;
  j["inclusive_ms"] = static_cast<double>(node->inclusive_ns) / 1'000'000.0;
  j["exclusive_ms"] = static_cast<double>(node->exclusive_ns) / 1'000'000.0;

  nlohmann::json children = nlohmann::json::array();
  for (const auto &child : node->children) {
    children.push_back(serialize_node(child.get()));
  }
  j["children"] = std::move(children);

  return j;
}

nlohmann::json Profiler::finish_run(std::int64_t run_total_ns) {
  nlohmann::json profile;
  profile["run_total_ms"] = static_cast<double>(run_total_ns) / 1'000'000.0;

  nlohmann::json frames = nlohmann::json::array();
  for (const auto &tree : frame_trees_) {
    frames.push_back(serialize_node(tree.get()));
  }
  profile["frames"] = std::move(frames);

  return profile;
}

} // namespace profiler
} // namespace moonai

namespace {

struct Args {
  int frames = 1000;
  std::vector<std::uint64_t> seeds = {61, 62, 63, 64, 65, 66};
  std::string output_dir = "output/profiles";
  std::string experiment_name = "profile";
  bool no_gpu = false;
};

Args parse_args(int argc, const char *argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--frames" && i + 1 < argc) {
      args.frames = std::stoi(argv[++i]);
    } else if (arg == "--output-dir" && i + 1 < argc) {
      args.output_dir = argv[++i];
    } else if (arg == "--name" && i + 1 < argc) {
      args.experiment_name = argv[++i];
    } else if (arg == "--no-gpu") {
      args.no_gpu = true;
    }
  }
  return args;
}

struct RunResult {
  std::uint64_t seed = 0;
  nlohmann::json profile;
  bool completed = false;
};

RunResult run_profiler(const moonai::AppConfig &cfg) {
  using namespace moonai;

  auto &profiler = profiler::Profiler::instance();

  profiler.start_run(cfg);

  const auto run_start = std::chrono::steady_clock::now();

  App app(cfg);
  bool completed = app.run();

  RunResult result;
  result.seed = cfg.sim_config.seed;

  if (!completed) {
    spdlog::info("Run stopped early, skipping profile generation");
    result.completed = false;
    return result;
  }

  const auto run_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - run_start)
                          .count();
  result.profile = profiler.finish_run(run_ns);
  result.completed = true;

  return result;
}

void write_manifest(std::vector<RunResult> runs,
                    const std::filesystem::path &output_path,
                    const moonai::AppConfig &cfg) {
  nlohmann::json manifest;
  manifest["schema_version"] = 1;

  nlohmann::json metadata;
  metadata["suite_name"] = cfg.experiment_name;
  metadata["frame_count"] = cfg.sim_config.max_steps;
  metadata["report_interval_steps"] = cfg.sim_config.report_interval_steps;
  metadata["gpu_allowed"] = cfg.enable_gpu;
  metadata["platform"] = moonai::AppConfig::platform;
  metadata["cuda_compiled"] = moonai::AppConfig::cuda_compiled;
  metadata["openmp_compiled"] = moonai::AppConfig::openmp_compiled;
  metadata["generated_at_utc"] = utc_timestamp();

  manifest["metadata"] = metadata;

  nlohmann::json run_rows = nlohmann::json::array();
  for (const auto &run : runs) {
    if (run.completed) {
      nlohmann::json run_obj;
      run_obj["seed"] = run.seed;
      run_obj["run_total_ms"] = run.profile["run_total_ms"];
      run_obj["frames"] = run.profile["frames"];
      run_rows.push_back(run_obj);
    }
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
  else
    spdlog::info("Profiler output written to: {}", output_path.string());
}

} // namespace

int main(int argc, const char *argv[]) {
  const Args args = parse_args(argc, argv);

  moonai::AppConfig base_cfg;
  base_cfg.sim_config = moonai::SimulationConfig();
  base_cfg.sim_config.max_steps = args.frames;
  base_cfg.experiment_name = args.experiment_name;
  base_cfg.headless = false;
  base_cfg.enable_gpu = !args.no_gpu;
  base_cfg.interactive = false;
  base_cfg.speed_multiplier = 1;
  const auto output_path =
      std::filesystem::path(args.output_dir) /
      (utc_timestamp() + "_" + args.experiment_name + ".json");

  std::vector<RunResult> runs;
  runs.reserve(args.seeds.size());
  for (std::uint64_t seed : args.seeds) {
    base_cfg.sim_config.seed = seed;
    runs.push_back(run_profiler(base_cfg));
  }

  write_manifest(std::move(runs), output_path, base_cfg);
  return 0;
}
