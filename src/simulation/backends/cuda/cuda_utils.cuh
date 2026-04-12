#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "simulation/backends/cuda/gpu_types.hpp"

// Log the CUDA error but continue
#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = (call);                                                                                          \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                       \
    }                                                                                                                  \
  } while (0)

namespace moonai::gpu {} // namespace moonai::gpu
