#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::FoodBuffer;
using moonai_gpu::FoodGridEntry;
using moonai_gpu::SimulationCounters;

namespace {

__global__ void resolve_food_kernel(DevicePopulationBuffers prey, FoodBuffer food, SimulationCounters *counters,
                                    const std::uint32_t *food_cell_offsets, const FoodGridEntry *food_entries,
                                    std::uint32_t *food_claimed_by, std::uint32_t grid_cols,
                                    std::uint32_t grid_rows, float grid_cell_size, float interaction_range) {
  const auto prey_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (prey_idx >= prey.capacity || prey.alive[prey_idx] == 0U) {
    return;
  }

  const auto interaction_range_sq = interaction_range * interaction_range;
  const auto px = prey.pos_x[prey_idx];
  const auto py = prey.pos_y[prey_idx];
  const auto cells_to_check = static_cast<std::int32_t>(interaction_range / grid_cell_size) + 1;
  const auto base_cx = static_cast<std::int32_t>(moonai_gpu::cell_coord(px, grid_cell_size, grid_cols));
  const auto base_cy = static_cast<std::int32_t>(moonai_gpu::cell_coord(py, grid_cell_size, grid_rows));

  std::uint32_t best_food = food.capacity;
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
      for (auto slot = food_cell_offsets[cell]; slot < food_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = food_entries[slot];
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto distance_sq = (dx * dx) + (dy * dy);
        if (distance_sq > best_distance_sq) {
          continue;
        }
        best_distance_sq = distance_sq;
        best_food = entry.slot;
      }
    }
  }

  if (best_food < food.capacity) {
    atomicMin(&food_claimed_by[best_food], prey_idx);
  }

  static_cast<void>(counters);
}

__global__ void finalize_food_kernel(DevicePopulationBuffers prey, FoodBuffer food, SimulationCounters *counters,
                                     const std::uint32_t *food_claimed_by, float energy_gain_from_food,
                                     float max_energy) {
  const auto food_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (food_idx >= food.capacity || food.active[food_idx] == 0U) {
    return;
  }

  const auto prey_idx = food_claimed_by[food_idx];
  if (prey_idx == moonai_gpu::kUnclaimedMate || prey_idx >= prey.capacity || prey.alive[prey_idx] == 0U) {
    return;
  }

  food.active[food_idx] = 0U;
  prey.energy[prey_idx] += energy_gain_from_food;
  if (prey.energy[prey_idx] > max_energy) {
    prey.energy[prey_idx] = max_energy;
  }
  atomicAdd(&counters->food_eaten, 1U);
}

__global__ void respawn_food_kernel(FoodBuffer food, const SimulationCounters *counters, std::uint64_t base_seed,
                                    float respawn_rate, float world_size) {
  const auto food_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (food_idx >= food.capacity || food.active[food_idx] != 0U) {
    return;
  }

  auto rng = moonai_gpu::splitmix64(base_seed ^ (static_cast<std::uint64_t>(counters->tick) << 32U) ^ food_idx);
  if (moonai_gpu::next_unit_float(rng) >= respawn_rate) {
    return;
  }

  food.pos_x[food_idx] = moonai_gpu::next_unit_float(rng) * world_size;
  food.pos_y[food_idx] = moonai_gpu::next_unit_float(rng) * world_size;
  food.active[food_idx] = 1U;
}

} // namespace

extern "C" std::int32_t dev_resolve_food_claims(DeviceState *state) {
  if (state->prey.capacity == 0U || state->food.capacity == 0U) {
    return 0;
  }
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  resolve_food_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, state->food, state->counters, state->food_cell_offsets, state->food_grid_entries,
      state->food_claimed_by, state->grid_cols, state->grid_rows, state->grid_cell_size,
      state->simulation.interaction_range);
  return moonai_gpu::launch_status();
}

extern "C" std::int32_t dev_finalize_food(DeviceState *state) {
  if (state->prey.capacity == 0U || state->food.capacity == 0U) {
    return 0;
  }
  const auto food_blocks = (state->food.capacity + 255U) / 256U;
  finalize_food_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(
      state->prey, state->food, state->counters, state->food_claimed_by, state->simulation.energy_gain_from_food,
      state->simulation.max_energy);
  return moonai_gpu::launch_status();
}

extern "C" std::int32_t dev_respawn_food(DeviceState *state) {
  if (state->food.capacity == 0U) {
    return 0;
  }
  const auto food_blocks = (state->food.capacity + 255U) / 256U;
  respawn_food_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(
      state->food, state->counters, state->simulation.seed ^ 0xC0FFEEULL, state->simulation.food_respawn_rate,
      state->simulation.grid_size);
  return moonai_gpu::launch_status();
}
