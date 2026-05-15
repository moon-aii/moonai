#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::FoodGridEntry;
using moonai_gpu::PopulationGridEntry;
using moonai_gpu::PopulationKind;
using moonai_gpu::SensorSnapshotReadback;

namespace {

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
  static_cast<void>(moonai_gpu::compute_sensor_inputs_for_slot<SelfIsPredator>(
      self_population, idx, predator_cell_offsets, predator_entries, prey_cell_offsets, prey_entries, food_cell_offsets,
      food_entries, grid_cols, grid_rows, grid_cell_size, num_inputs, vision_range, max_energy, agent_speed,
      world_size, out));
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
  return moonai_gpu::launch_status();
}

extern "C" std::int32_t dev_write_sensor_snapshot(const DeviceState *state, PopulationKind population_kind,
                                                   std::uint32_t slot) {
  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  sensor_snapshot_kernel<<<1U, 1U>>>(population, population_kind, slot, state->num_inputs,
                                     state->sensor_snapshot_scratch);
  return moonai_gpu::synchronize_kernels();
}
