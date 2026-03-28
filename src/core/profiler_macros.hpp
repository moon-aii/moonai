#pragma once

#include <cassert>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

// CUDA forward declaration for profiler (always defined)
#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#else
#include <cuda_runtime.h>
#endif

// Profiler macros - minimal header for use in other translation units
#ifndef MOONAI_BUILD_PROFILER

#define MOONAI_PROFILE_SCOPE(event_name, ...) ((void)0)

#else

namespace moonai {
namespace profiler {

// Forward declarations
class Profiler;
struct ScopeNode;

class ScopedTimer {
public:
  explicit ScopedTimer(const char *event_name, cudaStream_t stream = nullptr);
  ~ScopedTimer();

private:
  const char *event_name_;
  cudaStream_t stream_;
  int gpu_event_index_; // -1 if no GPU event
  bool has_cpu_scope_;  // false when stream provided (GPU-only mode)
};

} // namespace profiler
} // namespace moonai

#define MOONAI_PROFILE_SCOPE(event_name, ...)                                  \
  ::moonai::profiler::ScopedTimer _moonai_scoped_timer(event_name,             \
                                                       ##__VA_ARGS__)

#endif