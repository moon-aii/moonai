#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::PopulationGridEntry;
using moonai_gpu::PopulationKind;
using moonai_gpu::ReproductionPairReadback;
using moonai_gpu::SimulationCounters;

namespace {

__global__ void reset_reproduction_state_kernel(std::uint32_t *mate_claims, std::uint32_t capacity,
                                                std::uint32_t *out_pair_count) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < capacity) {
    mate_claims[idx] = moonai_gpu::kUnclaimedMate;
  }
  if (idx == 0U) {
    *out_pair_count = 0U;
  }
}

__global__ void find_reproduction_pairs_kernel(DevicePopulationBuffers population, const std::uint32_t *cell_offsets,
                                               const PopulationGridEntry *entries, std::uint32_t grid_cols,
                                               std::uint32_t grid_rows, float grid_cell_size, float mate_range,
                                               float reproduction_energy_threshold, std::uint32_t *mate_claims,
                                               ReproductionPairReadback *out_pairs, std::uint32_t *out_pair_count) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }
  if (population.energy[idx] < reproduction_energy_threshold) {
    return;
  }

  const auto px = population.pos_x[idx];
  const auto py = population.pos_y[idx];
  const auto mate_range_sq = mate_range * mate_range;
  const auto cells_to_check = static_cast<std::int32_t>(mate_range / grid_cell_size) + 1;
  const auto base_cx = static_cast<std::int32_t>(moonai_gpu::cell_coord(px, grid_cell_size, grid_cols));
  const auto base_cy = static_cast<std::int32_t>(moonai_gpu::cell_coord(py, grid_cell_size, grid_rows));

  std::uint32_t best_mate = population.capacity;
  float best_distance_sq = mate_range_sq;
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
                                                 grid_cell_size, px, py, mate_range)) {
        continue;
      }

      const auto cell = (static_cast<std::uint32_t>(cy) * grid_cols) + static_cast<std::uint32_t>(cx);
      for (auto slot = cell_offsets[cell]; slot < cell_offsets[cell + 1U]; ++slot) {
        const auto entry = entries[slot];
        if (entry.slot <= idx || population.alive[entry.slot] == 0U ||
            population.energy[entry.slot] < reproduction_energy_threshold) {
          continue;
        }

        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto dist_sq = (dx * dx) + (dy * dy);
        if (dist_sq > best_distance_sq || dist_sq <= 0.0F) {
          continue;
        }
        best_distance_sq = dist_sq;
        best_mate = entry.slot;
      }
    }
  }

  if (best_mate == population.capacity) {
    return;
  }

  if (atomicCAS(&mate_claims[best_mate], moonai_gpu::kUnclaimedMate, idx) != moonai_gpu::kUnclaimedMate) {
    return;
  }

  const auto pair_index = atomicAdd(out_pair_count, 1U);
  out_pairs[pair_index] = ReproductionPairReadback{idx, best_mate};
}

__global__ void prepare_reproduction_free_len_kernel(std::uint32_t *free_len, std::uint32_t free_slot_base) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *free_len = free_slot_base;
  }
}

__global__ void accumulate_reproduction_costs_kernel(const ReproductionPairReadback *pairs, std::uint32_t pair_count,
                                                     std::uint32_t *reproduction_counts,
                                                     std::uint32_t *birth_counter) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= pair_count) {
    return;
  }

  const auto pair = pairs[idx];
  atomicAdd(&reproduction_counts[pair.parent_a_slot], 1U);
  atomicAdd(&reproduction_counts[pair.parent_b_slot], 1U);
  atomicAdd(birth_counter, 1U);
}

__global__ void apply_reproduction_costs_kernel(DevicePopulationBuffers population,
                                                const std::uint32_t *reproduction_counts,
                                                float energy_cost, std::uint32_t *death_counter,
                                                std::uint32_t *free_list, std::uint32_t *free_len) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  const auto reproduction_count = reproduction_counts[idx];
  if (reproduction_count == 0U) {
    return;
  }

  const auto remaining_energy = population.energy[idx] - (energy_cost * static_cast<float>(reproduction_count));
  if (remaining_energy > 0.0F) {
    population.energy[idx] = remaining_energy;
    return;
  }

  population.energy[idx] = 0.0F;
  population.alive[idx] = 0U;
  population.vel_x[idx] = 0.0F;
  population.vel_y[idx] = 0.0F;
  const auto free_index = atomicAdd(free_len, 1U);
  if (free_index < population.capacity) {
    free_list[free_index] = idx;
  }
  atomicAdd(death_counter, 1U);
}

} // namespace

extern "C" std::int32_t dev_reset_population_reproduction_state(DeviceState *state, PopulationKind population_kind) {
  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *mate_claims = population_kind == PopulationKind::Predator ? state->predator_mate_claims : state->prey_mate_claims;
  auto *pair_count = population_kind == PopulationKind::Predator ? state->predator_pair_count : state->prey_pair_count;
  const auto blocks = (population.capacity + 255U) / 256U;
  reset_reproduction_state_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(mate_claims, population.capacity, pair_count);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_find_reproduction_pairs(DeviceState *state, PopulationKind population_kind) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *mate_claims = population_kind == PopulationKind::Predator ? state->predator_mate_claims : state->prey_mate_claims;
  auto *pair_buffer =
      population_kind == PopulationKind::Predator ? state->predator_reproduction_pairs : state->prey_reproduction_pairs;
  auto *pair_count_ptr = population_kind == PopulationKind::Predator ? state->predator_pair_count : state->prey_pair_count;
  auto *cell_offsets = population_kind == PopulationKind::Predator ? state->predator_cell_offsets : state->prey_cell_offsets;
  auto *entries = population_kind == PopulationKind::Predator ? state->predator_grid_entries : state->prey_grid_entries;
  const auto blocks = (population.capacity + 255U) / 256U;
  find_reproduction_pairs_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(
      population, cell_offsets, entries, state->grid_cols, state->grid_rows, state->grid_cell_size,
      state->simulation.mate_range, state->simulation.reproduction_energy_threshold, mate_claims, pair_buffer,
      pair_count_ptr);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_apply_reproduction_energy_kernel(DeviceState *state, PopulationKind population_kind,
                                                              std::uint32_t births_applied,
                                                              std::uint32_t free_slot_base) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  if (births_applied == 0U) {
    return 0;
  }

  auto *reproduction_counts = population_kind == PopulationKind::Predator ? state->predator_mate_claims
                                                                          : state->prey_mate_claims;
  const auto reset_status =
      cudaMemset(reproduction_counts, 0, static_cast<std::size_t>(population.capacity) * sizeof(std::uint32_t));
  if (reset_status != cudaSuccess) {
    return static_cast<std::int32_t>(reset_status);
  }

  auto *pair_buffer =
      population_kind == PopulationKind::Predator ? state->predator_reproduction_pairs : state->prey_reproduction_pairs;
  auto *birth_counter =
      population_kind == PopulationKind::Predator ? &state->counters->predator_births : &state->counters->prey_births;
  auto *death_counter =
      population_kind == PopulationKind::Predator ? &state->counters->predator_deaths : &state->counters->prey_deaths;
  auto *free_list = population_kind == PopulationKind::Predator ? state->predator_free_list : state->prey_free_list;
  auto *free_len = population_kind == PopulationKind::Predator ? state->predator_free_len : state->prey_free_len;
  const auto pair_blocks = (births_applied + 255U) / 256U;
  const auto population_blocks = (population.capacity + 255U) / 256U;
  accumulate_reproduction_costs_kernel<<<pair_blocks == 0U ? 1U : pair_blocks, 256U>>>(
      pair_buffer, births_applied, reproduction_counts, birth_counter);
  // Births consume slots from the tail so the remaining free-list prefix stays valid.
  prepare_reproduction_free_len_kernel<<<1U, 1U>>>(free_len, free_slot_base);
  apply_reproduction_costs_kernel<<<population_blocks == 0U ? 1U : population_blocks, 256U>>>(
      population, reproduction_counts, state->simulation.reproduction_energy_cost, death_counter, free_list,
      free_len);
  return moonai_gpu::synchronize_kernels();
}
