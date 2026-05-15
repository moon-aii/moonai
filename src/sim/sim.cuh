#pragma once

#include <cuda_runtime.h>
#include "moonai_gpu_ffi.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace moonai_gpu {

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

inline uint32_t synchronize_kernels() { 
  return cudaDeviceSynchronize();
}

inline uint32_t launch_status() {
  return cudaPeekAtLastError();
}

inline const DevicePopulationBuffers &population_for_kind(const DeviceState &state, PopulationKind population_kind) {
  return population_kind == PopulationKind::Predator ? state.predator : state.prey;
}

inline DevicePopulationBuffers &population_for_kind(DeviceState &state, PopulationKind population_kind) {
  return population_kind == PopulationKind::Predator ? state.predator : state.prey;
}

__device__ inline std::uint32_t cell_coord(float pos, float cell_size, std::uint32_t limit) {
  const auto coord = static_cast<std::uint32_t>(pos / cell_size);
  return coord < limit ? coord : limit - 1U;
}

__device__ inline bool cell_may_intersect_radius(std::uint32_t cx, std::uint32_t cy, float cell_size,
                                                 float origin_x, float origin_y, float radius) {
  const auto center_x = (static_cast<float>(cx) + 0.5F) * cell_size;
  const auto center_y = (static_cast<float>(cy) + 0.5F) * cell_size;
  const auto dx = center_x - origin_x;
  const auto dy = center_y - origin_y;
  const auto half_size = cell_size * 0.5F;
  const auto nearest_x = fmaxf(fabsf(dx) - half_size, 0.0F);
  const auto nearest_y = fmaxf(fabsf(dy) - half_size, 0.0F);
  return (nearest_x * nearest_x) + (nearest_y * nearest_y) <= radius * radius;
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
