#pragma once

#include <chrono>

// Profiler macros - minimal header for use in other translation units
#ifndef MOONAI_BUILD_PROFILER

#define MOONAI_PROFILE_SCOPE(event_name) ((void)0)
#define MOONAI_PROFILE_MARK_CPU_USED(value) ((void)0)
#define MOONAI_PROFILE_MARK_GPU_USED(value) ((void)0)

#else

namespace moonai {
namespace profiler {

class Profiler;

void mark_cpu_used(bool used);
void mark_gpu_used(bool used);

class ScopedTimer {
public:
  explicit ScopedTimer(const char *event_name);
  ~ScopedTimer();

private:
  const char *event_name_;
  bool active_;
  std::chrono::steady_clock::time_point start_;
};

} // namespace profiler
} // namespace moonai

#define MOONAI_PROFILE_SCOPE(event_name)                                       \
  ::moonai::profiler::ScopedTimer _moonai_scoped_timer(event_name)
#define MOONAI_PROFILE_MARK_CPU_USED(value)                                    \
  ::moonai::profiler::mark_cpu_used(value)
#define MOONAI_PROFILE_MARK_GPU_USED(value)                                    \
  ::moonai::profiler::mark_gpu_used(value)

#endif
