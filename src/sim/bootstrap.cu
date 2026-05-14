#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::FoodBuffer;
using moonai_gpu::PopulationKind;
using moonai_gpu::SimulationCounters;

namespace {

__global__ void seed_population_kernel(DevicePopulationBuffers population, std::uint32_t live_count,
                                       std::uint32_t base_entity_id, std::uint64_t base_seed,
                                       std::uint32_t num_inputs, std::uint32_t num_outputs, float world_size,
                                       float initial_energy, float max_energy, PopulationKind population_kind) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || idx >= live_count) {
    return;
  }

  std::uint64_t rng = base_seed ^
                      (static_cast<std::uint64_t>(population_kind == PopulationKind::Predator ? 0xA51U : 0xB29U)
                       << 32U) ^
                      idx;
  rng = moonai_gpu::splitmix64(rng);

  population.pos_x[idx] = moonai_gpu::next_unit_float(rng) * world_size;
  population.pos_y[idx] = moonai_gpu::next_unit_float(rng) * world_size;
  population.vel_x[idx] = 0.0F;
  population.vel_y[idx] = 0.0F;
  population.energy[idx] = initial_energy;
  population.age[idx] = 0.0F;
  population.alive[idx] = 1U;
  population.species_id[idx] = 0U;
  population.entity_id[idx] = base_entity_id + idx;
  population.generation[idx] = 0U;
  population.rng_state[idx] = rng;

  const auto sensor_base = static_cast<std::size_t>(idx) * num_inputs;
  for (std::uint32_t sensor = 0; sensor < num_inputs; ++sensor) {
    population.sensor_inputs[sensor_base + sensor] = 0.0F;
  }

  const auto seeded_node_count = static_cast<std::uint16_t>(num_inputs + num_outputs + 1U);
  const auto seeded_connection_count = static_cast<std::uint16_t>((num_inputs + 1U) * num_outputs);
  population.genome.num_nodes[idx] = seeded_node_count;
  population.genome.num_connections[idx] = seeded_connection_count;

  const auto node_base = static_cast<std::size_t>(idx) * population.genome.node_stride;
  for (std::uint32_t node = 0; node < population.genome.node_stride; ++node) {
    population.genome.node_types[node_base + node] = moonai_gpu::kInputNodeType;
  }
  for (std::uint32_t node = 0; node < num_inputs; ++node) {
    population.genome.node_types[node_base + node] = moonai_gpu::kInputNodeType;
  }
  population.genome.node_types[node_base + num_inputs] = moonai_gpu::kBiasNodeType;
  for (std::uint32_t node = 0; node < num_outputs; ++node) {
    population.genome.node_types[node_base + num_inputs + 1U + node] = moonai_gpu::kOutputNodeType;
  }

  const auto connection_base = static_cast<std::size_t>(idx) * population.genome.connection_stride;
  for (std::uint32_t connection = 0; connection < population.genome.connection_stride; ++connection) {
    population.genome.connection_from[connection_base + connection] = 0;
    population.genome.connection_to[connection_base + connection] = 0;
    population.genome.connection_weight[connection_base + connection] = 0.0F;
    population.genome.connection_innovation[connection_base + connection] = 0U;
    population.genome.connection_enabled[connection_base + connection] = 0U;
  }

  std::uint32_t connection_idx = 0U;
  for (std::uint32_t in_node = 0; in_node < num_inputs + 1U; ++in_node) {
    for (std::uint32_t out_idx = 0; out_idx < num_outputs; ++out_idx) {
      const auto slot = connection_base + connection_idx;
      population.genome.connection_from[slot] = static_cast<std::int32_t>(in_node);
      population.genome.connection_to[slot] = static_cast<std::int32_t>(num_inputs + 1U + out_idx);
      population.genome.connection_weight[slot] = moonai_gpu::next_signed_float(rng);
      population.genome.connection_innovation[slot] = (in_node * num_outputs) + out_idx;
      population.genome.connection_enabled[slot] = 1U;
      ++connection_idx;
    }
  }

  const auto compiled_node_base = static_cast<std::size_t>(idx) * population.compiled.node_stride;
  const auto compiled_connection_base = static_cast<std::size_t>(idx) * population.compiled.connection_stride;
  const auto compiled_output_base = static_cast<std::size_t>(idx) * population.compiled.output_stride;
  const auto compiled_offset_base = static_cast<std::size_t>(idx) * (population.compiled.node_stride + 1U);
  for (std::uint32_t node = 0; node < population.compiled.node_stride; ++node) {
    population.compiled.eval_order[compiled_node_base + node] = 0U;
    population.compiled.connection_offsets[compiled_offset_base + node] = 0U;
  }
  population.compiled.connection_offsets[compiled_offset_base + population.compiled.node_stride] = 0U;
  for (std::uint32_t output = 0; output < population.compiled.output_stride; ++output) {
    population.compiled.output_indices[compiled_output_base + output] = 0U;
  }
  for (std::uint32_t connection = 0; connection < population.compiled.connection_stride; ++connection) {
    population.compiled.connection_sources[compiled_connection_base + connection] = 0U;
    population.compiled.connection_weights[compiled_connection_base + connection] = 0.0F;
  }
  population.compiled.node_counts[idx] = seeded_node_count;
  population.compiled.eval_counts[idx] = 0U;
  population.compiled.connection_counts[idx] = 0U;

  static_cast<void>(max_energy);
}

__global__ void seed_food_kernel(FoodBuffer food, std::uint64_t base_seed, float world_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= food.capacity) {
    return;
  }

  auto rng = moonai_gpu::splitmix64(base_seed ^ (static_cast<std::uint64_t>(idx) << 16U));
  food.pos_x[idx] = moonai_gpu::next_unit_float(rng) * world_size;
  food.pos_y[idx] = moonai_gpu::next_unit_float(rng) * world_size;
  food.active[idx] = 1U;
}

__global__ void initialize_free_list_kernel(DevicePopulationBuffers population, std::uint32_t *free_list,
                                            std::uint32_t *free_len) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  std::uint32_t len = 0U;
  for (std::uint32_t idx = 0; idx < population.capacity; ++idx) {
    if (population.alive[idx] == 0U) {
      free_list[len++] = idx;
    }
  }
  *free_len = len;
}

__global__ void reset_simulation_counters_kernel(SimulationCounters *counters) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *counters = SimulationCounters{};
  }
}

} // namespace

extern "C" std::int32_t dev_seed_initial_population(DeviceState *state) {
  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  seed_population_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, state->simulation.predator_count, 1U, state->simulation.seed, state->num_inputs,
      state->num_outputs, state->simulation.grid_size, state->simulation.initial_energy, state->simulation.max_energy,
      PopulationKind::Predator);
  seed_population_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, state->simulation.prey_count, state->simulation.predator_count + 1U,
      state->simulation.seed ^ 0x9e3779b97f4a7c15ULL, state->num_inputs, state->num_outputs,
      state->simulation.grid_size, state->simulation.initial_energy, state->simulation.max_energy,
      PopulationKind::Prey);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_reset_counters(DeviceState *state) {
  reset_simulation_counters_kernel<<<1U, 1U>>>(state->counters);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_initialize_population_free_list(DeviceState *state, PopulationKind population_kind) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *free_list = population_kind == PopulationKind::Predator ? state->predator_free_list : state->prey_free_list;
  auto *free_len = population_kind == PopulationKind::Predator ? state->predator_free_len : state->prey_free_len;
  initialize_free_list_kernel<<<1U, 1U>>>(population, free_list, free_len);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_seed_food(DeviceState *state) {
  const auto food_blocks = (state->food.capacity + 255U) / 256U;
  seed_food_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(
      state->food, state->simulation.seed ^ 0xC0FFEEULL, state->simulation.grid_size);
  return moonai_gpu::synchronize_kernels();
}
