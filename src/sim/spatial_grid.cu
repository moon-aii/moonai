#include "sim.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::FoodBuffer;
using moonai_gpu::FoodGridEntry;
using moonai_gpu::PopulationGridEntry;
using moonai_gpu::PopulationKind;

namespace {

__global__ void count_population_cells_kernel(DevicePopulationBuffers population, std::uint32_t *cell_counts,
                                              std::uint32_t grid_cols, std::uint32_t grid_rows, float cell_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  const auto cx = moonai_gpu::cell_coord(population.pos_x[idx], cell_size, grid_cols);
  const auto cy = moonai_gpu::cell_coord(population.pos_y[idx], cell_size, grid_rows);
  atomicAdd(&cell_counts[(cy * grid_cols) + cx], 1U);
}

__global__ void scatter_population_cells_kernel(DevicePopulationBuffers population, std::uint32_t *cell_write_offsets,
                                                PopulationGridEntry *entries, std::uint32_t grid_cols,
                                                std::uint32_t grid_rows, float cell_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  const auto cx = moonai_gpu::cell_coord(population.pos_x[idx], cell_size, grid_cols);
  const auto cy = moonai_gpu::cell_coord(population.pos_y[idx], cell_size, grid_rows);
  const auto slot = atomicAdd(&cell_write_offsets[(cy * grid_cols) + cx], 1U);
  entries[slot] = PopulationGridEntry{idx, population.pos_x[idx], population.pos_y[idx]};
}

__global__ void count_food_cells_kernel(FoodBuffer food, std::uint32_t *cell_counts, std::uint32_t grid_cols,
                                        std::uint32_t grid_rows, float cell_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= food.capacity || food.active[idx] == 0U) {
    return;
  }

  const auto cx = moonai_gpu::cell_coord(food.pos_x[idx], cell_size, grid_cols);
  const auto cy = moonai_gpu::cell_coord(food.pos_y[idx], cell_size, grid_rows);
  atomicAdd(&cell_counts[(cy * grid_cols) + cx], 1U);
}

__global__ void scatter_food_cells_kernel(FoodBuffer food, std::uint32_t *cell_write_offsets, FoodGridEntry *entries,
                                          std::uint32_t grid_cols, std::uint32_t grid_rows, float cell_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= food.capacity || food.active[idx] == 0U) {
    return;
  }

  const auto cx = moonai_gpu::cell_coord(food.pos_x[idx], cell_size, grid_cols);
  const auto cy = moonai_gpu::cell_coord(food.pos_y[idx], cell_size, grid_rows);
  const auto slot = atomicAdd(&cell_write_offsets[(cy * grid_cols) + cx], 1U);
  entries[slot] = FoodGridEntry{idx, food.pos_x[idx], food.pos_y[idx]};
}

__global__ void finalize_cell_offsets_kernel(const std::uint32_t *cell_counts, std::uint32_t *cell_offsets,
                                             std::uint32_t *cell_write_offsets, std::uint32_t cell_count) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < cell_count) {
    cell_write_offsets[idx] = cell_offsets[idx];
  }
  if (idx == 0U) {
    cell_offsets[cell_count] = cell_count == 0U ? 0U : cell_offsets[cell_count - 1U] + cell_counts[cell_count - 1U];
  }
}

} // namespace

extern "C" std::int32_t dev_count_population_cells(const DeviceState *state, PopulationKind population_kind) {
  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *cell_counts = population_kind == PopulationKind::Predator ? state->predator_cell_counts : state->prey_cell_counts;
  const auto blocks = (population.capacity + 255U) / 256U;
  count_population_cells_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, cell_counts, state->grid_cols,
                                                                       state->grid_rows, state->grid_cell_size);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_count_food_cells(const DeviceState *state) {
  const auto food_blocks = (state->food.capacity + 255U) / 256U;
  count_food_cells_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(state->food, state->food_cell_counts,
                                                                            state->grid_cols, state->grid_rows,
                                                                            state->grid_cell_size);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_exclusive_scan_u32(const std::uint32_t *input, std::uint32_t count,
                                                std::uint32_t *output) {
  if (count == 0U) {
    return 0;
  }

  thrust::exclusive_scan(thrust::device, input, input + count, output);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_finalize_population_cell_offsets(const DeviceState *state, PopulationKind population_kind,
                                                             std::uint32_t cell_count) {
  auto *cell_counts = population_kind == PopulationKind::Predator ? state->predator_cell_counts : state->prey_cell_counts;
  auto *cell_offsets = population_kind == PopulationKind::Predator ? state->predator_cell_offsets : state->prey_cell_offsets;
  auto *cell_write_offsets =
      population_kind == PopulationKind::Predator ? state->predator_cell_write_offsets : state->prey_cell_write_offsets;
  const auto cell_blocks = (cell_count + 255U) / 256U;
  finalize_cell_offsets_kernel<<<cell_blocks == 0U ? 1U : cell_blocks, 256U>>>(cell_counts, cell_offsets,
                                                                                 cell_write_offsets, cell_count);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_finalize_food_cell_offsets(const DeviceState *state, std::uint32_t cell_count) {
  const auto cell_blocks = (cell_count + 255U) / 256U;
  finalize_cell_offsets_kernel<<<cell_blocks == 0U ? 1U : cell_blocks, 256U>>>(state->food_cell_counts,
                                                                                 state->food_cell_offsets,
                                                                                 state->food_cell_write_offsets,
                                                                                 cell_count);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_scatter_population_cells(const DeviceState *state, PopulationKind population_kind) {
  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *cell_write_offsets =
      population_kind == PopulationKind::Predator ? state->predator_cell_write_offsets : state->prey_cell_write_offsets;
  auto *grid_entries = population_kind == PopulationKind::Predator ? state->predator_grid_entries : state->prey_grid_entries;
  const auto blocks = (population.capacity + 255U) / 256U;
  scatter_population_cells_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, cell_write_offsets, grid_entries,
                                                                         state->grid_cols, state->grid_rows,
                                                                         state->grid_cell_size);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_scatter_food_cells(const DeviceState *state) {
  const auto food_blocks = (state->food.capacity + 255U) / 256U;
  scatter_food_cells_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(state->food,
                                                                              state->food_cell_write_offsets,
                                                                              state->food_grid_entries,
                                                                              state->grid_cols, state->grid_rows,
                                                                              state->grid_cell_size);
  return moonai_gpu::synchronize_kernels();
}
