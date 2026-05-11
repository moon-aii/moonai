#pragma once

#include <cassert>

// CUDA forward declaration for profiler
#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#else
#include <cuda_runtime.h>
#endif

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
  int stream_event_index_; // -1 if no stream event
  bool has_cpu_scope_;     // false when stream provided (stream-only mode)
};

} // namespace profiler
} // namespace moonai

#define MOONAI_PROFILE_SCOPE(event_name, ...)                                                                          \
  ::moonai::profiler::ScopedTimer _moonai_scoped_timer(event_name, ##__VA_ARGS__)

#endif
