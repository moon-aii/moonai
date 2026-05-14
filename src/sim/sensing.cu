#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::FoodGridEntry;
using moonai_gpu::PopulationGridEntry;
using moonai_gpu::PopulationKind;
using moonai_gpu::SensorSnapshotReadback;

namespace {

__device__ float clamp_range(float value, float min_value, float max_value) {
  return fminf(fmaxf(value, min_value), max_value);
}

template <std::uint32_t N>
__device__ void insert_nearest_candidate(float dx, float dy, float dist_sq, float (&best_dx)[N],
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
__device__ void encode_nearest_targets(const float (&best_dx)[N], const float (&best_dy)[N],
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

__device__ float encode_axis_wall_sensor(float negative_side_dist, float positive_side_dist, float vision_range) {
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

template <bool SelfIsPredator>
__global__ void compute_sensor_inputs_kernel(DevicePopulationBuffers self_population,
                                             const std::uint32_t *predator_cell_offsets,
                                             const PopulationGridEntry *predator_entries,
                                             const std::uint32_t *prey_cell_offsets,
                                             const PopulationGridEntry *prey_entries,
                                             const std::uint32_t *food_cell_offsets,
                                             const FoodGridEntry *food_entries, std::uint32_t grid_cols,
                                             std::uint32_t grid_rows, float grid_cell_size,
                                             std::uint32_t num_inputs, float vision_range, float max_energy,
                                             float agent_speed, float world_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= self_population.capacity || self_population.sensor_inputs == nullptr) {
    return;
  }

  const auto sensor_base = static_cast<std::size_t>(idx) * num_inputs;
  auto *out = self_population.sensor_inputs + sensor_base;
  for (std::uint32_t sensor_idx = 0; sensor_idx < num_inputs; ++sensor_idx) {
    out[sensor_idx] = 0.0F;
  }

  if (self_population.alive[idx] == 0U) {
    return;
  }

  const auto px = self_population.pos_x[idx];
  const auto py = self_population.pos_y[idx];
  const auto vision_sq = vision_range * vision_range;
  const auto cells_to_check = static_cast<std::int32_t>(vision_range / grid_cell_size) + 1;
  const auto base_cx = static_cast<std::int32_t>(moonai_gpu::cell_coord(px, grid_cell_size, grid_cols));
  const auto base_cy = static_cast<std::int32_t>(moonai_gpu::cell_coord(py, grid_cell_size, grid_rows));
  float predator_dx[moonai_gpu::kNearestTargetsPerType];
  float predator_dy[moonai_gpu::kNearestTargetsPerType];
  float predator_dist_sq[moonai_gpu::kNearestTargetsPerType];
  float prey_dx[moonai_gpu::kNearestTargetsPerType];
  float prey_dy[moonai_gpu::kNearestTargetsPerType];
  float prey_dist_sq[moonai_gpu::kNearestTargetsPerType];
  float food_dx[moonai_gpu::kNearestTargetsPerType];
  float food_dy[moonai_gpu::kNearestTargetsPerType];
  float food_dist_sq[moonai_gpu::kNearestTargetsPerType];
  for (std::uint32_t nearest_idx = 0; nearest_idx < moonai_gpu::kNearestTargetsPerType; ++nearest_idx) {
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
      if (!moonai_gpu::cell_may_intersect_radius(static_cast<std::uint32_t>(cx), static_cast<std::uint32_t>(cy),
                                                 grid_cell_size, px, py, vision_range)) {
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

  if (num_inputs > 0U) {
    encode_nearest_targets(predator_dx, predator_dy, predator_dist_sq, vision_range, out,
                           num_inputs < moonai_gpu::kPerTypeSensorCount ? num_inputs : moonai_gpu::kPerTypeSensorCount);
  }
  if (num_inputs > moonai_gpu::kPerTypeSensorCount) {
    const auto prey_output_capacity = num_inputs - moonai_gpu::kPerTypeSensorCount;
    encode_nearest_targets(prey_dx, prey_dy, prey_dist_sq, vision_range, out + moonai_gpu::kPerTypeSensorCount,
                           prey_output_capacity < moonai_gpu::kPerTypeSensorCount ? prey_output_capacity
                                                                                   : moonai_gpu::kPerTypeSensorCount);
  }
  if (num_inputs > (2U * moonai_gpu::kPerTypeSensorCount)) {
    const auto food_output_capacity = num_inputs - (2U * moonai_gpu::kPerTypeSensorCount);
    encode_nearest_targets(food_dx, food_dy, food_dist_sq, vision_range, out + (2U * moonai_gpu::kPerTypeSensorCount),
                           food_output_capacity < moonai_gpu::kPerTypeSensorCount ? food_output_capacity
                                                                                   : moonai_gpu::kPerTypeSensorCount);
  }
  if (num_inputs > moonai_gpu::kSelfEnergyInputIndex) {
    out[moonai_gpu::kSelfEnergyInputIndex] =
        max_energy <= 0.0F ? 0.0F : clamp_range(self_population.energy[idx] / max_energy, 0.0F, 1.0F);
  }
  if (agent_speed > 0.0F && num_inputs > moonai_gpu::kVelocityXInputIndex) {
    out[moonai_gpu::kVelocityXInputIndex] = clamp_range(self_population.vel_x[idx] / agent_speed, -1.0F, 1.0F);
  }
  if (agent_speed > 0.0F && num_inputs > moonai_gpu::kVelocityYInputIndex) {
    out[moonai_gpu::kVelocityYInputIndex] = clamp_range(self_population.vel_y[idx] / agent_speed, -1.0F, 1.0F);
  }
  if (num_inputs > moonai_gpu::kWallXInputIndex) {
    out[moonai_gpu::kWallXInputIndex] = encode_axis_wall_sensor(px, world_size - px, vision_range);
  }
  if (num_inputs > moonai_gpu::kWallYInputIndex) {
    out[moonai_gpu::kWallYInputIndex] = encode_axis_wall_sensor(py, world_size - py, vision_range);
  }
}

__global__ void sensor_snapshot_kernel(DevicePopulationBuffers population, PopulationKind population_kind,
                                       std::uint32_t slot, std::uint32_t num_inputs,
                                       SensorSnapshotReadback *out_snapshot) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  out_snapshot->population_kind = population_kind;
  out_snapshot->slot = slot;
  out_snapshot->input_count =
      static_cast<std::uint16_t>(num_inputs < moonai_gpu::kSensorInputCount ? num_inputs : moonai_gpu::kSensorInputCount);
  out_snapshot->reserved = 0U;
  for (std::uint32_t sensor_idx = 0; sensor_idx < moonai_gpu::kSensorInputCount; ++sensor_idx) {
    out_snapshot->inputs[sensor_idx] = 0.0F;
  }

  if (population.sensor_inputs == nullptr) {
    return;
  }

  const auto sensor_base = static_cast<std::size_t>(slot) * num_inputs;
  for (std::uint32_t sensor_idx = 0; sensor_idx < out_snapshot->input_count; ++sensor_idx) {
    out_snapshot->inputs[sensor_idx] = population.sensor_inputs[sensor_base + sensor_idx];
  }
}

} // namespace

extern "C" std::int32_t dev_compute_sensor_inputs(DeviceState *state) {
  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  compute_sensor_inputs_kernel<true><<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.predator_speed, state->simulation.grid_size);
  compute_sensor_inputs_kernel<false><<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.prey_speed, state->simulation.grid_size);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t dev_write_sensor_snapshot(const DeviceState *state, PopulationKind population_kind,
                                                   std::uint32_t slot) {
  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  sensor_snapshot_kernel<<<1U, 1U>>>(population, population_kind, slot, state->num_inputs,
                                     state->sensor_snapshot_scratch);
  return moonai_gpu::synchronize_kernels();
}
