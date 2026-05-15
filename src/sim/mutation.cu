#include "sim.cuh"

using moonai_gpu::DeviceInnovationState;
using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::GpuMutationConfig;
using moonai_gpu::PopulationKind;

namespace {

__device__ bool node_can_emit(std::uint8_t node_type) { return node_type != moonai_gpu::kOutputNodeType; }

__device__ bool node_can_receive(std::uint8_t node_type) {
  return node_type == moonai_gpu::kHiddenNodeType || node_type == moonai_gpu::kOutputNodeType;
}

__device__ bool connection_exists(const DevicePopulationBuffers &population, std::uint32_t slot,
                                  std::uint16_t connection_count, std::int32_t from_node, std::int32_t to_node) {
  const auto connection_base = static_cast<std::size_t>(slot) * population.genome.connection_stride;
  for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
    const auto entry = connection_base + connection;
    if (population.genome.connection_from[entry] == from_node && population.genome.connection_to[entry] == to_node) {
      return true;
    }
  }
  return false;
}

__device__ int random_enabled_connection(const DevicePopulationBuffers &population, std::uint32_t slot,
                                         std::uint16_t connection_count, std::uint64_t &rng) {
  if (connection_count == 0U) {
    return -1;
  }

  const auto connection_base = static_cast<std::size_t>(slot) * population.genome.connection_stride;
  for (std::uint16_t attempt = 0; attempt < connection_count; ++attempt) {
    const auto candidate = static_cast<std::uint16_t>(moonai_gpu::next_unit_float(rng) * connection_count) % connection_count;
    if (population.genome.connection_enabled[connection_base + candidate] != 0U) {
      return static_cast<int>(candidate);
    }
  }

  for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
    if (population.genome.connection_enabled[connection_base + connection] != 0U) {
      return static_cast<int>(connection);
    }
  }
  return -1;
}

__device__ bool choose_connection_endpoints(const DevicePopulationBuffers &population, std::uint32_t slot,
                                            std::uint16_t node_count, std::uint32_t max_attempts,
                                            std::uint64_t &rng, std::int32_t &from_node, std::int32_t &to_node) {
  const auto node_base = static_cast<std::size_t>(slot) * population.genome.node_stride;
  for (std::uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
    const auto from_candidate = static_cast<std::uint16_t>(moonai_gpu::next_unit_float(rng) * node_count) % node_count;
    const auto to_candidate = static_cast<std::uint16_t>(moonai_gpu::next_unit_float(rng) * node_count) % node_count;
    if (from_candidate == to_candidate) {
      continue;
    }

    const auto from_type = population.genome.node_types[node_base + from_candidate];
    const auto to_type = population.genome.node_types[node_base + to_candidate];
    if (!node_can_emit(from_type) || !node_can_receive(to_type)) {
      continue;
    }
    if (from_type == moonai_gpu::kHiddenNodeType && to_type == moonai_gpu::kHiddenNodeType && from_candidate >= to_candidate) {
      continue;
    }

    from_node = static_cast<std::int32_t>(from_candidate);
    to_node = static_cast<std::int32_t>(to_candidate);
    return true;
  }

  return false;
}

__device__ void mutate_single_agent(DevicePopulationBuffers population, std::uint32_t idx, DeviceInnovationState *innovation,
                                    GpuMutationConfig config) {
  auto rng = population.rng_state[idx];
  auto connection_count = population.genome.num_connections[idx];
  auto node_count = population.genome.num_nodes[idx];
  const auto connection_base = static_cast<std::size_t>(idx) * population.genome.connection_stride;
  const auto node_base = static_cast<std::size_t>(idx) * population.genome.node_stride;

  if (moonai_gpu::next_unit_float(rng) < config.mutation_rate) {
    for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
      const auto entry = connection_base + connection;
      if (population.genome.connection_enabled[entry] == 0U) {
        continue;
      }
      if (moonai_gpu::next_unit_float(rng) >= config.mutation_rate) {
        continue;
      }
      population.genome.connection_weight[entry] += moonai_gpu::next_signed_float(rng) * config.weight_mutation_power;
    }
  }

  if (moonai_gpu::next_unit_float(rng) < config.delete_connection_rate) {
    const auto candidate = random_enabled_connection(population, idx, connection_count, rng);
    if (candidate >= 0) {
      population.genome.connection_enabled[connection_base + static_cast<std::uint16_t>(candidate)] = 0U;
    }
  }

  if (moonai_gpu::next_unit_float(rng) < config.add_connection_rate && connection_count < population.genome.connection_stride) {
    std::int32_t from_node = 0;
    std::int32_t to_node = 0;
    if (choose_connection_endpoints(population, idx, node_count, config.max_connection_attempts, rng, from_node, to_node) &&
        !connection_exists(population, idx, connection_count, from_node, to_node)) {
      const auto innovation_id = atomicAdd(&innovation->next_innovation, 1U);
      const auto entry = connection_base + connection_count;
      population.genome.connection_from[entry] = from_node;
      population.genome.connection_to[entry] = to_node;
      population.genome.connection_weight[entry] = moonai_gpu::next_signed_float(rng);
      population.genome.connection_innovation[entry] = innovation_id;
      population.genome.connection_enabled[entry] = 1U;
      ++connection_count;
    }
  }

  if (moonai_gpu::next_unit_float(rng) < config.add_node_rate && node_count < population.genome.node_stride &&
      connection_count + 1U < population.genome.connection_stride) {
    const auto split_connection = random_enabled_connection(population, idx, connection_count, rng);
    if (split_connection >= 0) {
      const auto source_entry = connection_base + static_cast<std::uint16_t>(split_connection);
      const auto source_from = population.genome.connection_from[source_entry];
      const auto source_to = population.genome.connection_to[source_entry];
      const auto source_weight = population.genome.connection_weight[source_entry];
      population.genome.connection_enabled[source_entry] = 0U;

      const auto new_node_index = node_count;
      population.genome.node_types[node_base + new_node_index] = moonai_gpu::kHiddenNodeType;
      population.genome.num_nodes[idx] = static_cast<std::uint16_t>(node_count + 1U);
      node_count = static_cast<std::uint16_t>(node_count + 1U);

      static_cast<void>(atomicAdd(&innovation->next_node_id, 1U));
      const auto innovation_a = atomicAdd(&innovation->next_innovation, 1U);
      const auto innovation_b = atomicAdd(&innovation->next_innovation, 1U);

      const auto first_entry = connection_base + connection_count;
      population.genome.connection_from[first_entry] = source_from;
      population.genome.connection_to[first_entry] = static_cast<std::int32_t>(new_node_index);
      population.genome.connection_weight[first_entry] = 1.0F;
      population.genome.connection_innovation[first_entry] = innovation_a;
      population.genome.connection_enabled[first_entry] = 1U;

      const auto second_entry = connection_base + connection_count + 1U;
      population.genome.connection_from[second_entry] = static_cast<std::int32_t>(new_node_index);
      population.genome.connection_to[second_entry] = source_to;
      population.genome.connection_weight[second_entry] = source_weight;
      population.genome.connection_innovation[second_entry] = innovation_b;
      population.genome.connection_enabled[second_entry] = 1U;
      connection_count = static_cast<std::uint16_t>(connection_count + 2U);
    }
  }

  population.genome.num_connections[idx] = connection_count;
  population.rng_state[idx] = rng;
  population.compiled.eval_counts[idx] = 0U;
  population.compiled.connection_counts[idx] = 0U;
}

__global__ void mutate_single_slot_kernel(DevicePopulationBuffers population, DeviceInnovationState *innovation, GpuMutationConfig config, std::uint32_t slot) {
  if (blockIdx.x != 0U || threadIdx.x != 0U || slot >= population.capacity || population.alive[slot] == 0U) {
    return;
  }

  mutate_single_agent(population, slot, innovation, config);
}

__global__ void mutate_batch_kernel(DevicePopulationBuffers population, DeviceInnovationState *innovation,
                                    GpuMutationConfig config, const std::uint32_t *free_list,
                                    std::uint32_t free_slot_base, std::uint32_t births_applied) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= births_applied) {
    return;
  }

  mutate_single_agent(population, free_list[free_slot_base + idx], innovation, config);
}

} // namespace

extern "C" std::int32_t dev_mutate_slot(DeviceState *state, PopulationKind population_kind, uint32_t slot, const GpuMutationConfig *config) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);

  mutate_single_slot_kernel<<<1U, 1U>>>(population, state->innovation, *config, slot);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_mutate_batch(DeviceState *state, PopulationKind population_kind,
                                          std::uint32_t births_applied, std::uint32_t free_slot_base,
                                          const GpuMutationConfig *config) {
  if (births_applied == 0U) {
    return 0;
  }

  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *free_list = population_kind == PopulationKind::Predator ? state->predator_free_list : state->prey_free_list;
  const auto blocks = (births_applied + 255U) / 256U;
  mutate_batch_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, state->innovation, *config, free_list,
                                                             free_slot_base, births_applied);
  return moonai_gpu::synchronize_kernels();
}
