#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::PopulationKind;
using moonai_gpu::SimulationCounters;

namespace {

__device__ float clamp_world(float value, float world_size) {
  if (value < 0.0F) {
    return 0.0F;
  }
  if (value > world_size) {
    return world_size;
  }
  return value;
}

__global__ void advance_tick_kernel(SimulationCounters *counters) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    ++counters->tick;
  }
}

__global__ void infer_population_kernel(DevicePopulationBuffers population, std::uint32_t num_inputs) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  float activations[moonai_gpu::kCompileScratchNodeLimit]{};
  const auto node_count = moonai_gpu::evaluate_compiled_network(population, idx, num_inputs, activations);
  if (node_count == 0U) {
    population.vel_x[idx] = 0.0F;
    population.vel_y[idx] = 0.0F;
    return;
  }

  population.vel_x[idx] = moonai_gpu::compiled_output_activation(population, idx, node_count, activations, 0U);
  population.vel_y[idx] = moonai_gpu::compiled_output_activation(population, idx, node_count, activations, 1U);
}

__global__ void update_vitals_kernel(DevicePopulationBuffers population, float energy_drain_per_tick,
                                     std::uint32_t max_age, std::uint32_t *free_list, std::uint32_t *free_len,
                                     std::uint32_t *death_counter) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  population.age[idx] += 1.0F;
  population.energy[idx] -= energy_drain_per_tick;
  if (population.energy[idx] < 0.0F) {
    population.energy[idx] = 0.0F;
  }

  if (population.energy[idx] <= 0.0F || population.age[idx] >= static_cast<float>(max_age)) {
    population.alive[idx] = 0U;
    population.vel_x[idx] = 0.0F;
    population.vel_y[idx] = 0.0F;
    const auto free_index = atomicAdd(free_len, 1U);
    if (free_index < population.capacity) {
      free_list[free_index] = idx;
    }
    atomicAdd(death_counter, 1U);
  }
}

__global__ void apply_movement_kernel(DevicePopulationBuffers population, float speed, float world_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  population.pos_x[idx] = clamp_world(population.pos_x[idx] + (population.vel_x[idx] * speed), world_size);
  population.pos_y[idx] = clamp_world(population.pos_y[idx] + (population.vel_y[idx] * speed), world_size);
}

} // namespace

extern "C" std::int32_t dev_infer_population(DeviceState *state, PopulationKind population_kind) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  const auto blocks = (population.capacity + 255U) / 256U;
  infer_population_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, state->num_inputs);
  return static_cast<std::int32_t>(moonai_gpu::launch_status());
}

extern "C" std::int32_t dev_update_vitals(DeviceState *state, PopulationKind population_kind) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *free_list = population_kind == PopulationKind::Predator ? state->predator_free_list : state->prey_free_list;
  auto *free_len = population_kind == PopulationKind::Predator ? state->predator_free_len : state->prey_free_len;
  auto *death_counter =
      population_kind == PopulationKind::Predator ? &state->counters->predator_deaths : &state->counters->prey_deaths;
  const auto blocks = (population.capacity + 255U) / 256U;
  update_vitals_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, state->simulation.energy_drain_per_tick,
                                                              state->simulation.max_age, free_list, free_len,
                                                              death_counter);
  return static_cast<std::int32_t>(moonai_gpu::launch_status());
}

extern "C" std::int32_t dev_apply_movement(DeviceState *state, PopulationKind population_kind) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  const auto speed =
      population_kind == PopulationKind::Predator ? state->simulation.predator_speed : state->simulation.prey_speed;
  const auto blocks = (population.capacity + 255U) / 256U;
  apply_movement_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, speed, state->simulation.grid_size);
  return moonai_gpu::launch_status();
}

extern "C" std::int32_t dev_advance_tick(DeviceState *state) {
  advance_tick_kernel<<<1U, 1U>>>(state->counters);
  return moonai_gpu::synchronize_kernels();
}
