#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "moonai_gpu_ffi.hpp"

namespace moonai_gpu {

extern std::int32_t g_last_cuda_error_code;

constexpr std::uint8_t kInputNodeType = 0;
constexpr std::uint8_t kHiddenNodeType = 1;
constexpr std::uint8_t kOutputNodeType = 2;
constexpr std::uint8_t kBiasNodeType = 3;
constexpr std::uint32_t kCompileScratchNodeLimit = 128U;
constexpr std::uint32_t kCompileScratchConnectionLimit = 256U;
constexpr std::uint32_t kSpeciesBucketCount = 64U;
constexpr std::uint32_t kNearestTargetsPerType = 5U;
constexpr std::uint32_t kTargetCoordinateCount = 2U;
constexpr std::uint32_t kPerTypeSensorCount = kNearestTargetsPerType * kTargetCoordinateCount;
constexpr std::uint32_t kTargetTypeCount = 3U;
constexpr std::uint32_t kTargetSensorCount = kPerTypeSensorCount * kTargetTypeCount;
constexpr std::uint32_t kSelfStateSensorCount = 3U;
constexpr std::uint32_t kWallSensorCount = 2U;
constexpr std::uint32_t kSensorInputCount = kTargetSensorCount + kSelfStateSensorCount + kWallSensorCount;
constexpr std::uint32_t kSelfEnergyInputIndex = kTargetSensorCount;
constexpr std::uint32_t kVelocityXInputIndex = kSelfEnergyInputIndex + 1U;
constexpr std::uint32_t kVelocityYInputIndex = kSelfEnergyInputIndex + 2U;
constexpr std::uint32_t kWallXInputIndex = kSelfEnergyInputIndex + kSelfStateSensorCount;
constexpr std::uint32_t kWallYInputIndex = kWallXInputIndex + 1U;
constexpr std::uint32_t kUnclaimedMate = 0xFFFF'FFFFU;

struct DeviceGenomeBuffers {
  std::int32_t *connection_from;
  std::int32_t *connection_to;
  float *connection_weight;
  std::uint32_t *connection_innovation;
  std::uint8_t *connection_enabled;
  std::uint8_t *node_types;
  std::uint16_t *num_connections;
  std::uint16_t *num_nodes;
  std::uint32_t connection_stride;
  std::uint32_t node_stride;
};

struct DeviceCompiledNetworkBuffers {
  std::uint16_t *eval_order;
  std::uint32_t *connection_offsets;
  std::uint16_t *output_indices;
  std::uint16_t *connection_sources;
  float *connection_weights;
  std::uint16_t *node_counts;
  std::uint16_t *eval_counts;
  std::uint16_t *connection_counts;
  std::uint32_t node_stride;
  std::uint32_t connection_stride;
  std::uint32_t output_stride;
};

struct DevicePopulationBuffers {
  float *pos_x;
  float *pos_y;
  float *vel_x;
  float *vel_y;
  float *energy;
  float *age;
  std::uint8_t *alive;
  std::uint32_t *species_id;
  std::uint32_t *entity_id;
  std::uint32_t *generation;
  std::uint64_t *rng_state;
  float *sensor_inputs;
  DeviceGenomeBuffers genome;
  DeviceCompiledNetworkBuffers compiled;
  std::uint32_t capacity;
};

struct FoodBuffer {
  float *pos_x;
  float *pos_y;
  std::uint8_t *active;
  std::uint32_t capacity;
};

struct DeviceInnovationState {
  std::uint32_t next_innovation;
  std::uint32_t next_node_id;
};

struct SimulationCounters {
  std::uint32_t tick;
  std::uint32_t predator_births;
  std::uint32_t prey_births;
  std::uint32_t predator_deaths;
  std::uint32_t prey_deaths;
  std::uint32_t kills;
  std::uint32_t food_eaten;
};

struct PopulationGridEntry {
  std::uint32_t slot;
  float pos_x;
  float pos_y;
};

struct FoodGridEntry {
  std::uint32_t slot;
  float pos_x;
  float pos_y;
};

struct GpuEvolutionState {
  GpuEvolutionConfig config;
  SimulationConfig simulation;
  DevicePopulationBuffers predator;
  DevicePopulationBuffers prey;
  FoodBuffer food;
  DeviceInnovationState *innovation;
  std::uint32_t *next_entity_id;
  SimulationCounters *counters;
  std::uint32_t *predator_free_list;
  std::uint32_t *prey_free_list;
  std::uint32_t *predator_free_len;
  std::uint32_t *prey_free_len;
  std::uint32_t *predator_mate_claims;
  std::uint32_t *prey_mate_claims;
  ReproductionPairReadback *predator_reproduction_pairs;
  ReproductionPairReadback *prey_reproduction_pairs;
  std::uint32_t *predator_pair_count;
  std::uint32_t *prey_pair_count;
  std::uint32_t *population_live_count_scratch;
  UiStatsReadback *ui_stats_scratch;
  FreeListStateReadback *free_list_state_scratch;
  SensorSnapshotReadback *sensor_snapshot_scratch;
  CompiledNetworkReadbackHeader *compiled_header_scratch;
  SelectedAgentNetworkReadback *selected_network_scratch;
  MetricsSummaryReadback *metrics_summary;
  SpeciesSummaryReadback *species_summaries_scratch;
  RepresentativeGenomeHeader *representative_headers_scratch;
  std::uint32_t *species_count_scratch;
  RenderSnapshotHeader *render_header_scratch;
  RenderAgentReadback *render_predators_scratch;
  RenderAgentReadback *render_prey_scratch;
  RenderFoodReadback *render_food_scratch;
  std::uint32_t *predator_cell_counts;
  std::uint32_t *predator_cell_offsets;
  std::uint32_t *predator_cell_write_offsets;
  PopulationGridEntry *predator_grid_entries;
  std::uint32_t *prey_cell_counts;
  std::uint32_t *prey_cell_offsets;
  std::uint32_t *prey_cell_write_offsets;
  PopulationGridEntry *prey_grid_entries;
  std::uint32_t *food_cell_counts;
  std::uint32_t *food_cell_offsets;
  std::uint32_t *food_cell_write_offsets;
  FoodGridEntry *food_grid_entries;
  std::uint32_t *food_claimed_by;
  std::uint32_t *prey_claimed_by;
  std::uint32_t grid_cols;
  std::uint32_t grid_rows;
  std::uint32_t grid_cell_capacity;
  float grid_cell_size;
};

inline bool is_runtime_unavailable_error(cudaError_t error) {
  return error == cudaErrorInsufficientDriver || error == cudaErrorInitializationError || error == cudaErrorNoDevice;
}

inline CudaStatus map_cuda_runtime_error(cudaError_t error, CudaStatus failure_status) {
  g_last_cuda_error_code = static_cast<std::int32_t>(error);
  if (is_runtime_unavailable_error(error)) {
    return CudaStatus::RuntimeUnavailable;
  }
  return error == cudaSuccess ? CudaStatus::Success : failure_status;
}

template <typename T> inline CudaStatus alloc_array(T **ptr, std::size_t count) {
  if (count == 0U) {
    *ptr = nullptr;
    return CudaStatus::Success;
  }

  void *raw = nullptr;
  const auto error = cudaMalloc(&raw, count * sizeof(T));
  const auto status = map_cuda_runtime_error(error, CudaStatus::AllocationFailed);
  if (status != CudaStatus::Success) {
    return status;
  }
  *ptr = static_cast<T *>(raw);
  return CudaStatus::Success;
}

template <typename T> inline void free_array(T *&ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

inline CudaStatus zero_device_memory(void *ptr, std::size_t size) {
  return map_cuda_runtime_error(cudaMemset(ptr, 0, size), CudaStatus::DeviceCopyFailed);
}

inline CudaStatus copy_compact_device_readback(const void *device_ptr, void *host_ptr, std::size_t size) {
  return map_cuda_runtime_error(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost), CudaStatus::DeviceCopyFailed);
}

inline CudaStatus copy_host_data_to_device(void *device_ptr, const void *host_ptr, std::size_t size) {
  return map_cuda_runtime_error(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice), CudaStatus::DeviceCopyFailed);
}

inline CudaStatus synchronize_kernels() { return map_cuda_runtime_error(cudaDeviceSynchronize(), CudaStatus::KernelLaunchFailed); }

inline const DevicePopulationBuffers &population_for_kind(const GpuEvolutionState &state, PopulationKind population_kind) {
  return population_kind == PopulationKind::Predator ? state.predator : state.prey;
}

inline DevicePopulationBuffers &population_for_kind(GpuEvolutionState &state, PopulationKind population_kind) {
  return population_kind == PopulationKind::Predator ? state.predator : state.prey;
}

__device__ inline std::uint64_t splitmix64(std::uint64_t state) {
  state += 0x9e3779b97f4a7c15ULL;
  state = (state ^ (state >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  state = (state ^ (state >> 27U)) * 0x94d049bb133111ebULL;
  return state ^ (state >> 31U);
}

__device__ inline float next_unit_float(std::uint64_t &state) {
  state = splitmix64(state);
  const auto bits = static_cast<std::uint32_t>(state & 0x00FF'FFFFULL);
  return static_cast<float>(bits) / static_cast<float>(0x00FF'FFFFU);
}

__device__ inline float next_signed_float(std::uint64_t &state) { return (next_unit_float(state) * 2.0F) - 1.0F; }

__device__ inline std::uint64_t hash_mix(std::uint64_t hash, std::uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
  return hash;
}

__device__ inline std::uint16_t count_enabled_connections(const DevicePopulationBuffers &population,
                                                          std::uint32_t slot,
                                                          std::uint16_t connection_count) {
  const auto connection_base = static_cast<std::size_t>(slot) * population.genome.connection_stride;
  std::uint16_t enabled_count = 0U;
  for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
    enabled_count = static_cast<std::uint16_t>(enabled_count +
                                               (population.genome.connection_enabled[connection_base + connection] != 0U));
  }
  return enabled_count;
}

__device__ inline std::uint16_t evaluate_compiled_network(const DevicePopulationBuffers &population,
                                                          std::uint32_t slot, std::uint32_t num_inputs,
                                                          float (&activations)[kCompileScratchNodeLimit]) {
  const auto node_count = population.compiled.node_counts[slot];
  if (node_count == 0U || node_count > kCompileScratchNodeLimit) {
    return 0U;
  }

  if (population.sensor_inputs != nullptr) {
    const auto sensor_base = static_cast<std::size_t>(slot) * num_inputs;
    for (std::uint32_t input = 0; input < num_inputs && input < node_count; ++input) {
      activations[input] = population.sensor_inputs[sensor_base + input];
    }
  }
  if (num_inputs < node_count) {
    activations[num_inputs] = 1.0F;
  }

  const auto eval_count = population.compiled.eval_counts[slot];
  const auto offset_base = static_cast<std::size_t>(slot) * (population.compiled.node_stride + 1U);
  const auto eval_base = static_cast<std::size_t>(slot) * population.compiled.node_stride;
  const auto compiled_connection_base = static_cast<std::size_t>(slot) * population.compiled.connection_stride;
  for (std::uint16_t eval_index = 0; eval_index < eval_count; ++eval_index) {
    const auto node = population.compiled.eval_order[eval_base + eval_index];
    if (node >= node_count) {
      continue;
    }
    const auto start = population.compiled.connection_offsets[offset_base + node];
    const auto end = population.compiled.connection_offsets[offset_base + node + 1U];
    float sum = 0.0F;
    for (std::uint32_t connection = start; connection < end; ++connection) {
      const auto source = population.compiled.connection_sources[compiled_connection_base + connection];
      if (source < node_count) {
        sum += activations[source] * population.compiled.connection_weights[compiled_connection_base + connection];
      }
    }
    activations[node] = tanhf(sum);
  }

  return node_count;
}

__device__ inline float compiled_output_activation(const DevicePopulationBuffers &population, std::uint32_t slot,
                                                   std::uint16_t node_count,
                                                   const float (&activations)[kCompileScratchNodeLimit],
                                                   std::uint32_t output_index) {
  if (output_index >= population.compiled.output_stride) {
    return 0.0F;
  }
  const auto output_base = static_cast<std::size_t>(slot) * population.compiled.output_stride;
  const auto output_node = population.compiled.output_indices[output_base + output_index];
  return output_node < node_count ? activations[output_node] : 0.0F;
}

} // namespace moonai_gpu
