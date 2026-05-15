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

__device__ inline float clamp_range(float value, float min_value, float max_value) {
  return fminf(fmaxf(value, min_value), max_value);
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

template <std::uint32_t N>
__device__ inline void insert_nearest_candidate(float dx, float dy, float dist_sq, float (&best_dx)[N],
                                                float (&best_dy)[N], float (&best_dist_sq)[N]) {
  if (dist_sq >= best_dist_sq[N - 1U]) {
    return;
  }

  std::uint32_t insert_at = N - 1U;
  while (insert_at > 0U && dist_sq < best_dist_sq[insert_at - 1U]) {
    best_dist_sq[insert_at] = best_dist_sq[insert_at - 1U];
    best_dx[insert_at] = best_dx[insert_at - 1U];
    best_dy[insert_at] = best_dy[insert_at - 1U];
    --insert_at;
  }

  best_dist_sq[insert_at] = dist_sq;
  best_dx[insert_at] = dx;
  best_dy[insert_at] = dy;
}

template <std::uint32_t N>
__device__ inline void encode_nearest_targets(const float (&best_dx)[N], const float (&best_dy)[N],
                                              const float (&best_dist_sq)[N], float vision_range, float *out,
                                              std::uint32_t output_capacity) {
  for (std::uint32_t idx = 0; idx < N; ++idx) {
    const auto offset = idx * 2U;
    if (offset + 1U >= output_capacity) {
      return;
    }
    if (best_dist_sq[idx] == INFINITY) {
      out[offset] = 0.0F;
      out[offset + 1U] = 0.0F;
      continue;
    }

    const auto dist = sqrtf(best_dist_sq[idx]);
    if (dist <= 1e-6F) {
      out[offset] = 0.0F;
      out[offset + 1U] = 0.0F;
      continue;
    }

    const auto proximity = clamp_range(1.0F - (dist / vision_range), 0.0F, 1.0F);
    const auto inv_dist = 1.0F / dist;
    out[offset] = clamp_range(best_dx[idx] * inv_dist * proximity, -1.0F, 1.0F);
    out[offset + 1U] = clamp_range(best_dy[idx] * inv_dist * proximity, -1.0F, 1.0F);
  }
}

__device__ inline float encode_axis_wall_sensor(float negative_side_dist, float positive_side_dist,
                                                float vision_range) {
  const auto negative_in_range = negative_side_dist < vision_range;
  const auto positive_in_range = positive_side_dist < vision_range;
  if (!negative_in_range && !positive_in_range) {
    return 0.0F;
  }
  if (negative_in_range && (!positive_in_range || negative_side_dist <= positive_side_dist)) {
    return -(1.0F - (negative_side_dist / vision_range));
  }
  return 1.0F - (positive_side_dist / vision_range);
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

template <bool SelfIsPredator>
__device__ inline bool compute_sensor_inputs_for_slot(const DevicePopulationBuffers &self_population,
                                                      std::uint32_t idx,
                                                      const std::uint32_t *predator_cell_offsets,
                                                      const PopulationGridEntry *predator_entries,
                                                      const std::uint32_t *prey_cell_offsets,
                                                      const PopulationGridEntry *prey_entries,
                                                      const std::uint32_t *food_cell_offsets,
                                                      const FoodGridEntry *food_entries,
                                                      std::uint32_t grid_cols, std::uint32_t grid_rows,
                                                      float grid_cell_size, std::uint32_t num_inputs,
                                                      float vision_range, float max_energy, float agent_speed,
                                                      float world_size, float *out) {
  const auto output_count = num_inputs < kSensorInputCount ? num_inputs : kSensorInputCount;
  for (std::uint32_t sensor_idx = 0; sensor_idx < output_count; ++sensor_idx) {
    out[sensor_idx] = 0.0F;
  }

  if (idx >= self_population.capacity || self_population.alive[idx] == 0U) {
    return false;
  }

  const auto px = self_population.pos_x[idx];
  const auto py = self_population.pos_y[idx];
  const auto vision_sq = vision_range * vision_range;
  const auto cells_to_check = static_cast<std::int32_t>(vision_range / grid_cell_size) + 1;
  const auto base_cx = static_cast<std::int32_t>(cell_coord(px, grid_cell_size, grid_cols));
  const auto base_cy = static_cast<std::int32_t>(cell_coord(py, grid_cell_size, grid_rows));
  float predator_dx[kNearestTargetsPerType];
  float predator_dy[kNearestTargetsPerType];
  float predator_dist_sq[kNearestTargetsPerType];
  float prey_dx[kNearestTargetsPerType];
  float prey_dy[kNearestTargetsPerType];
  float prey_dist_sq[kNearestTargetsPerType];
  float food_dx[kNearestTargetsPerType];
  float food_dy[kNearestTargetsPerType];
  float food_dist_sq[kNearestTargetsPerType];
  for (std::uint32_t nearest_idx = 0; nearest_idx < kNearestTargetsPerType; ++nearest_idx) {
    predator_dx[nearest_idx] = 0.0F;
    predator_dy[nearest_idx] = 0.0F;
    predator_dist_sq[nearest_idx] = INFINITY;
    prey_dx[nearest_idx] = 0.0F;
    prey_dy[nearest_idx] = 0.0F;
    prey_dist_sq[nearest_idx] = INFINITY;
    food_dx[nearest_idx] = 0.0F;
    food_dy[nearest_idx] = 0.0F;
    food_dist_sq[nearest_idx] = INFINITY;
  }

  for (auto dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    const auto cy = base_cy + dy_cell;
    if (cy < 0 || cy >= static_cast<std::int32_t>(grid_rows)) {
      continue;
    }
    for (auto dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const auto cx = base_cx + dx_cell;
      if (cx < 0 || cx >= static_cast<std::int32_t>(grid_cols)) {
        continue;
      }
      if (!cell_may_intersect_radius(static_cast<std::uint32_t>(cx), static_cast<std::uint32_t>(cy), grid_cell_size,
                                     px, py, vision_range)) {
        continue;
      }

      const auto cell = (static_cast<std::uint32_t>(cy) * grid_cols) + static_cast<std::uint32_t>(cx);
      for (auto slot = predator_cell_offsets[cell]; slot < predator_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = predator_entries[slot];
        if (SelfIsPredator && entry.slot == idx) {
          continue;
        }
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto dist_sq = (dx * dx) + (dy * dy);
        if (dist_sq > vision_sq || dist_sq <= 0.0F) {
          continue;
        }
        insert_nearest_candidate(dx, dy, dist_sq, predator_dx, predator_dy, predator_dist_sq);
      }

      for (auto slot = prey_cell_offsets[cell]; slot < prey_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = prey_entries[slot];
        if (!SelfIsPredator && entry.slot == idx) {
          continue;
        }
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto dist_sq = (dx * dx) + (dy * dy);
        if (dist_sq > vision_sq || dist_sq <= 0.0F) {
          continue;
        }
        insert_nearest_candidate(dx, dy, dist_sq, prey_dx, prey_dy, prey_dist_sq);
      }

      for (auto slot = food_cell_offsets[cell]; slot < food_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = food_entries[slot];
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto dist_sq = (dx * dx) + (dy * dy);
        if (dist_sq > vision_sq || dist_sq <= 0.0F) {
          continue;
        }
        insert_nearest_candidate(dx, dy, dist_sq, food_dx, food_dy, food_dist_sq);
      }
    }
  }

  if (output_count > 0U) {
    encode_nearest_targets(predator_dx, predator_dy, predator_dist_sq, vision_range, out,
                           output_count < kPerTypeSensorCount ? output_count : kPerTypeSensorCount);
  }
  if (output_count > kPerTypeSensorCount) {
    const auto prey_output_capacity = output_count - kPerTypeSensorCount;
    encode_nearest_targets(prey_dx, prey_dy, prey_dist_sq, vision_range, out + kPerTypeSensorCount,
                           prey_output_capacity < kPerTypeSensorCount ? prey_output_capacity : kPerTypeSensorCount);
  }
  if (output_count > (2U * kPerTypeSensorCount)) {
    const auto food_output_capacity = output_count - (2U * kPerTypeSensorCount);
    encode_nearest_targets(food_dx, food_dy, food_dist_sq, vision_range, out + (2U * kPerTypeSensorCount),
                           food_output_capacity < kPerTypeSensorCount ? food_output_capacity : kPerTypeSensorCount);
  }
  if (output_count > kSelfEnergyInputIndex) {
    out[kSelfEnergyInputIndex] =
        max_energy <= 0.0F ? 0.0F : clamp_range(self_population.energy[idx] / max_energy, 0.0F, 1.0F);
  }
  if (agent_speed > 0.0F && output_count > kVelocityXInputIndex) {
    out[kVelocityXInputIndex] = clamp_range(self_population.vel_x[idx] / agent_speed, -1.0F, 1.0F);
  }
  if (agent_speed > 0.0F && output_count > kVelocityYInputIndex) {
    out[kVelocityYInputIndex] = clamp_range(self_population.vel_y[idx] / agent_speed, -1.0F, 1.0F);
  }
  if (output_count > kWallXInputIndex) {
    out[kWallXInputIndex] = encode_axis_wall_sensor(px, world_size - px, vision_range);
  }
  if (output_count > kWallYInputIndex) {
    out[kWallYInputIndex] = encode_axis_wall_sensor(py, world_size - py, vision_range);
  }

  return true;
}

__device__ inline std::uint16_t evaluate_compiled_network_seeded(const DevicePopulationBuffers &population,
                                                                 std::uint32_t slot, std::uint32_t num_inputs,
                                                                 float (&activations)[kCompileScratchNodeLimit]) {
  const auto node_count = population.compiled.node_counts[slot];
  if (node_count == 0U || node_count > kCompileScratchNodeLimit) {
    return 0U;
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

__device__ inline std::uint16_t evaluate_compiled_network(const DevicePopulationBuffers &population,
                                                          std::uint32_t slot, std::uint32_t num_inputs,
                                                          float (&activations)[kCompileScratchNodeLimit]) {
  if (population.sensor_inputs != nullptr) {
    const auto sensor_base = static_cast<std::size_t>(slot) * num_inputs;
    for (std::uint32_t input = 0; input < num_inputs && input < kCompileScratchNodeLimit; ++input) {
      activations[input] = population.sensor_inputs[sensor_base + input];
    }
  }
  return evaluate_compiled_network_seeded(population, slot, num_inputs, activations);
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
