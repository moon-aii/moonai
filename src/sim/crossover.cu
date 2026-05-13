#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::PopulationKind;

namespace {

__device__ std::uint8_t merge_node_type(std::uint8_t first, std::uint8_t second) {
  if (first == second) {
    return first;
  }
  if (first == moonai_gpu::kHiddenNodeType || second == moonai_gpu::kHiddenNodeType) {
    return moonai_gpu::kHiddenNodeType;
  }
  if (first == moonai_gpu::kBiasNodeType || second == moonai_gpu::kBiasNodeType) {
    return moonai_gpu::kBiasNodeType;
  }
  if (first == moonai_gpu::kOutputNodeType || second == moonai_gpu::kOutputNodeType) {
    return moonai_gpu::kOutputNodeType;
  }
  return moonai_gpu::kInputNodeType;
}

__device__ void write_connection_gene(const DevicePopulationBuffers &population, std::uint32_t slot,
                                      std::uint16_t connection_index, std::int32_t from_node, std::int32_t to_node,
                                      float weight, std::uint32_t innovation, std::uint8_t enabled) {
  const auto entry = static_cast<std::size_t>(slot) * population.genome.connection_stride + connection_index;
  population.genome.connection_from[entry] = from_node;
  population.genome.connection_to[entry] = to_node;
  population.genome.connection_weight[entry] = weight;
  population.genome.connection_innovation[entry] = innovation;
  population.genome.connection_enabled[entry] = enabled;
}

__global__ void crossover_kernel(DevicePopulationBuffers population, const std::uint32_t *next_entity_id,
                                 PopulationKind population_kind, std::uint32_t parent_a_slot,
                                 std::uint32_t parent_b_slot, std::uint32_t offspring_slot,
                                 float initial_energy) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  const auto parent_a_node_count = population.genome.num_nodes[parent_a_slot];
  const auto parent_b_node_count = population.genome.num_nodes[parent_b_slot];
  const auto parent_a_connection_count = population.genome.num_connections[parent_a_slot];
  const auto parent_b_connection_count = population.genome.num_connections[parent_b_slot];
  const auto child_node_count = parent_a_node_count > parent_b_node_count ? parent_a_node_count : parent_b_node_count;

  const auto parent_a_node_base = static_cast<std::size_t>(parent_a_slot) * population.genome.node_stride;
  const auto parent_b_node_base = static_cast<std::size_t>(parent_b_slot) * population.genome.node_stride;
  const auto child_node_base = static_cast<std::size_t>(offspring_slot) * population.genome.node_stride;
  for (std::uint32_t node = 0; node < population.genome.node_stride; ++node) {
    population.genome.node_types[child_node_base + node] = moonai_gpu::kInputNodeType;
  }
  for (std::uint16_t node = 0; node < child_node_count; ++node) {
    const auto first_type = node < parent_a_node_count ? population.genome.node_types[parent_a_node_base + node]
                                                       : moonai_gpu::kInputNodeType;
    const auto second_type = node < parent_b_node_count ? population.genome.node_types[parent_b_node_base + node]
                                                        : moonai_gpu::kInputNodeType;
    population.genome.node_types[child_node_base + node] = merge_node_type(first_type, second_type);
  }

  const auto parent_a_connection_base = static_cast<std::size_t>(parent_a_slot) * population.genome.connection_stride;
  const auto parent_b_connection_base = static_cast<std::size_t>(parent_b_slot) * population.genome.connection_stride;
  const auto child_connection_base = static_cast<std::size_t>(offspring_slot) * population.genome.connection_stride;
  for (std::uint32_t connection = 0; connection < population.genome.connection_stride; ++connection) {
    write_connection_gene(population, offspring_slot, static_cast<std::uint16_t>(connection), 0, 0, 0.0F, 0U, 0U);
  }

  auto rng = population.rng_state[parent_a_slot] ^ moonai_gpu::splitmix64(population.rng_state[parent_b_slot]) ^ offspring_slot;
  std::uint16_t parent_a_index = 0U;
  std::uint16_t parent_b_index = 0U;
  std::uint16_t child_connection_count = 0U;
  std::uint32_t matching_genes = 0U;
  std::uint32_t disjoint_genes = 0U;
  std::uint32_t excess_genes = 0U;

  while (child_connection_count < population.genome.connection_stride &&
         (parent_a_index < parent_a_connection_count || parent_b_index < parent_b_connection_count)) {
    const auto a_active = parent_a_index < parent_a_connection_count;
    const auto b_active = parent_b_index < parent_b_connection_count;

    if (a_active && b_active) {
      const auto a_entry = parent_a_connection_base + parent_a_index;
      const auto b_entry = parent_b_connection_base + parent_b_index;
      const auto a_innovation = population.genome.connection_innovation[a_entry];
      const auto b_innovation = population.genome.connection_innovation[b_entry];

      if (a_innovation == b_innovation) {
        ++matching_genes;
        const auto choose_parent_b = moonai_gpu::next_unit_float(rng) < 0.5F;
        const auto source_entry = choose_parent_b ? b_entry : a_entry;
        auto enabled = population.genome.connection_enabled[source_entry];
        if ((population.genome.connection_enabled[a_entry] == 0U || population.genome.connection_enabled[b_entry] == 0U) &&
            moonai_gpu::next_unit_float(rng) < 0.75F) {
          enabled = 0U;
        }
        write_connection_gene(population, offspring_slot, child_connection_count,
                              population.genome.connection_from[source_entry], population.genome.connection_to[source_entry],
                              population.genome.connection_weight[source_entry],
                              population.genome.connection_innovation[source_entry], enabled);
        ++child_connection_count;
        ++parent_a_index;
        ++parent_b_index;
        continue;
      }

      const auto inherit_from_a = a_innovation < b_innovation;
      const auto source_entry = inherit_from_a ? a_entry : b_entry;
      ++disjoint_genes;
      if (moonai_gpu::next_unit_float(rng) < 0.5F) {
        const auto enabled = moonai_gpu::next_unit_float(rng) < 0.75F ? 0U : population.genome.connection_enabled[source_entry];
        write_connection_gene(population, offspring_slot, child_connection_count,
                              population.genome.connection_from[source_entry], population.genome.connection_to[source_entry],
                              population.genome.connection_weight[source_entry],
                              population.genome.connection_innovation[source_entry], enabled);
        ++child_connection_count;
      }
      if (inherit_from_a) {
        ++parent_a_index;
      } else {
        ++parent_b_index;
      }
      continue;
    }

    const auto inherit_from_a = a_active;
    const auto source_entry = inherit_from_a ? parent_a_connection_base + parent_a_index : parent_b_connection_base + parent_b_index;
    ++excess_genes;
    if (moonai_gpu::next_unit_float(rng) < 0.5F) {
      const auto enabled = moonai_gpu::next_unit_float(rng) < 0.75F ? 0U : population.genome.connection_enabled[source_entry];
      write_connection_gene(population, offspring_slot, child_connection_count,
                            population.genome.connection_from[source_entry], population.genome.connection_to[source_entry],
                            population.genome.connection_weight[source_entry], population.genome.connection_innovation[source_entry],
                            enabled);
      ++child_connection_count;
    }
    if (inherit_from_a) {
      ++parent_a_index;
    } else {
      ++parent_b_index;
    }
  }

  population.genome.num_nodes[offspring_slot] = child_node_count;
  population.genome.num_connections[offspring_slot] = child_connection_count;
  population.alive[offspring_slot] = 1U;
  population.species_id[offspring_slot] = 0U;
  population.entity_id[offspring_slot] = atomicAdd(const_cast<std::uint32_t *>(next_entity_id), 1U);
  population.generation[offspring_slot] =
      (population.generation[parent_a_slot] > population.generation[parent_b_slot] ? population.generation[parent_a_slot]
                                                                                    : population.generation[parent_b_slot]) +
      1U;
  population.pos_x[offspring_slot] = (population.pos_x[parent_a_slot] + population.pos_x[parent_b_slot]) * 0.5F;
  population.pos_y[offspring_slot] = (population.pos_y[parent_a_slot] + population.pos_y[parent_b_slot]) * 0.5F;
  population.vel_x[offspring_slot] = moonai_gpu::next_signed_float(rng) * 0.05F;
  population.vel_y[offspring_slot] = moonai_gpu::next_signed_float(rng) * 0.05F;
  population.energy[offspring_slot] = initial_energy;
  population.age[offspring_slot] = 0.0F;
  population.rng_state[offspring_slot] = rng;
  population.compiled.node_counts[offspring_slot] = child_node_count;
  population.compiled.eval_counts[offspring_slot] = 0U;
  population.compiled.connection_counts[offspring_slot] = 0U;

  std::uint64_t genome_hash = 1469598103934665603ULL;
  for (std::uint16_t node = 0; node < child_node_count; ++node) {
    genome_hash = moonai_gpu::hash_mix(genome_hash, population.genome.node_types[child_node_base + node]);
  }
  for (std::uint16_t connection = 0; connection < child_connection_count; ++connection) {
    const auto entry = child_connection_base + connection;
    genome_hash = moonai_gpu::hash_mix(genome_hash, static_cast<std::uint32_t>(population.genome.connection_from[entry]));
    genome_hash = moonai_gpu::hash_mix(genome_hash, static_cast<std::uint32_t>(population.genome.connection_to[entry]));
    genome_hash = moonai_gpu::hash_mix(genome_hash, population.genome.connection_innovation[entry]);
    genome_hash = moonai_gpu::hash_mix(genome_hash, population.genome.connection_enabled[entry]);
    genome_hash =
        moonai_gpu::hash_mix(genome_hash, static_cast<std::uint64_t>(__float_as_uint(population.genome.connection_weight[entry])));
  }

  static_cast<void>(population_kind);
  static_cast<void>(matching_genes);
  static_cast<void>(disjoint_genes);
  static_cast<void>(excess_genes);
  static_cast<void>(genome_hash);
}

} // namespace

extern "C" std::int32_t dev_crossover(DeviceState *state, PopulationKind population_kind, std::uint32_t parent_a_slot, std::uint32_t parent_b_slot, std::uint32_t offspring_slot) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  const auto offspring_energy = state->simulation.offspring_initial_energy > 0.0F ? state->simulation.offspring_initial_energy : state->config.initial_energy;
  crossover_kernel<<<1U, 1U>>>(population, state->next_entity_id, population_kind, parent_a_slot, parent_b_slot, offspring_slot, offspring_energy);
  return moonai_gpu::synchronize_kernels();
}
