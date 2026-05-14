#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::PopulationGridEntry;
using moonai_gpu::SimulationCounters;

namespace {

__global__ void resolve_combat_kernel(DevicePopulationBuffers predator, DevicePopulationBuffers prey,
                                      std::uint32_t *prey_claimed_by, const std::uint32_t *prey_cell_offsets,
                                      const PopulationGridEntry *prey_entries, std::uint32_t grid_cols,
                                      std::uint32_t grid_rows, float grid_cell_size, float interaction_range) {
  const auto predator_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (predator_idx >= predator.capacity || predator.alive[predator_idx] == 0U) {
    return;
  }

  const auto interaction_range_sq = interaction_range * interaction_range;
  const auto px = predator.pos_x[predator_idx];
  const auto py = predator.pos_y[predator_idx];
  const auto cells_to_check = static_cast<std::int32_t>(interaction_range / grid_cell_size) + 1;
  const auto base_cx = static_cast<std::int32_t>(moonai_gpu::cell_coord(px, grid_cell_size, grid_cols));
  const auto base_cy = static_cast<std::int32_t>(moonai_gpu::cell_coord(py, grid_cell_size, grid_rows));

  std::uint32_t best_prey = prey.capacity;
  float best_distance_sq = interaction_range_sq;
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
                                                 grid_cell_size, px, py, interaction_range)) {
        continue;
      }

      const auto cell = (static_cast<std::uint32_t>(cy) * grid_cols) + static_cast<std::uint32_t>(cx);
      for (auto slot = prey_cell_offsets[cell]; slot < prey_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = prey_entries[slot];
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto distance_sq = (dx * dx) + (dy * dy);
        if (distance_sq > best_distance_sq) {
          continue;
        }
        best_distance_sq = distance_sq;
        best_prey = entry.slot;
      }
    }
  }

  if (best_prey < prey.capacity) {
    atomicMin(&prey_claimed_by[best_prey], predator_idx);
  }
}

__global__ void finalize_combat_kernel(DevicePopulationBuffers predator, DevicePopulationBuffers prey,
                                       SimulationCounters *counters, std::uint32_t *prey_free_list,
                                       std::uint32_t *prey_free_len, const std::uint32_t *prey_claimed_by,
                                       float energy_gain_from_kill, float max_energy) {
  const auto prey_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (prey_idx >= prey.capacity || prey.alive[prey_idx] == 0U) {
    return;
  }

  const auto predator_idx = prey_claimed_by[prey_idx];
  if (predator_idx == moonai_gpu::kUnclaimedMate || predator_idx >= predator.capacity ||
      predator.alive[predator_idx] == 0U) {
    return;
  }

  prey.alive[prey_idx] = 0U;
  prey.energy[prey_idx] = 0.0F;
  prey.vel_x[prey_idx] = 0.0F;
  prey.vel_y[prey_idx] = 0.0F;
  const auto free_index = atomicAdd(prey_free_len, 1U);
  if (free_index < prey.capacity) {
    prey_free_list[free_index] = prey_idx;
  }
  predator.energy[predator_idx] += energy_gain_from_kill;
  if (predator.energy[predator_idx] > max_energy) {
    predator.energy[predator_idx] = max_energy;
  }
  atomicAdd(&counters->kills, 1U);
  atomicAdd(&counters->prey_deaths, 1U);
}

} // namespace

extern "C" std::int32_t dev_resolve_combat_claims(DeviceState *state) {
  if (state->predator.capacity == 0U || state->prey.capacity == 0U) {
    return 0;
  }
  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  resolve_combat_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, state->prey, state->prey_claimed_by, state->prey_cell_offsets, state->prey_grid_entries,
      state->grid_cols, state->grid_rows, state->grid_cell_size, state->simulation.interaction_range);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_finalize_combat(DeviceState *state) {
  if (state->predator.capacity == 0U || state->prey.capacity == 0U) {
    return 0;
  }
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  finalize_combat_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->predator, state->prey, state->counters, state->prey_free_list, state->prey_free_len,
      state->prey_claimed_by, state->simulation.energy_gain_from_kill, state->simulation.max_energy);
  return moonai_gpu::synchronize_kernels();
}
