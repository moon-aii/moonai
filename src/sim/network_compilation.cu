#include "sim.cuh"

using moonai_gpu::CompiledNetworkReadbackHeader;
using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::PopulationKind;
using moonai_gpu::RepresentativeGenomeHeader;
using moonai_gpu::SelectedAgentNetworkReadback;
using moonai_gpu::SpeciesBatchReadbackHeader;
using moonai_gpu::SpeciesReduceScratch;
using moonai_gpu::SpeciesSummaryReadback;

namespace {

__device__ std::uint64_t genome_hash(const DevicePopulationBuffers &population, std::uint32_t slot,
                                     std::uint16_t node_count, std::uint16_t connection_count) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto node_base = static_cast<std::size_t>(slot) * population.genome.node_stride;
  const auto connection_base = static_cast<std::size_t>(slot) * population.genome.connection_stride;
  for (std::uint16_t node = 0; node < node_count; ++node) {
    hash = moonai_gpu::hash_mix(hash, population.genome.node_types[node_base + node]);
  }
  for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
    const auto entry = connection_base + connection;
    hash = moonai_gpu::hash_mix(hash, static_cast<std::uint32_t>(population.genome.connection_from[entry]));
    hash = moonai_gpu::hash_mix(hash, static_cast<std::uint32_t>(population.genome.connection_to[entry]));
    hash = moonai_gpu::hash_mix(hash, population.genome.connection_innovation[entry]);
    hash = moonai_gpu::hash_mix(hash, population.genome.connection_enabled[entry]);
    hash = moonai_gpu::hash_mix(hash, static_cast<std::uint64_t>(__float_as_uint(population.genome.connection_weight[entry])));
  }
  return hash;
}

__device__ std::uint32_t species_bucket(const DevicePopulationBuffers &population, std::uint32_t slot,
                                        std::uint16_t node_count, std::uint16_t connection_count) {
  const auto enabled_count = moonai_gpu::count_enabled_connections(population, slot, connection_count);
  const auto complexity = static_cast<std::uint32_t>(node_count) + static_cast<std::uint32_t>(enabled_count);
  const auto hash = genome_hash(population, slot, node_count, connection_count);
  return (complexity + static_cast<std::uint32_t>(hash & 31U)) % moonai_gpu::kSpeciesBucketCount;
}

__device__ void build_representative_header(const DevicePopulationBuffers &population, PopulationKind population_kind,
                                            std::uint32_t slot, RepresentativeGenomeHeader *out_header) {
  out_header->population_kind = population_kind;
  out_header->slot = slot;
  out_header->entity_id = population.entity_id[slot];
  out_header->generation = population.generation[slot];
  out_header->species_id = population.species_id[slot];
  out_header->num_nodes = population.genome.num_nodes[slot];
  out_header->num_connections = population.genome.num_connections[slot];
}

__device__ void compile_slot_device(DevicePopulationBuffers population, std::uint32_t idx, std::uint32_t output_stride) {
  if (population.alive[idx] == 0U) {
    population.compiled.node_counts[idx] = 0U;
    population.compiled.eval_counts[idx] = 0U;
    population.compiled.connection_counts[idx] = 0U;
    return;
  }

  const auto node_count = population.genome.num_nodes[idx];
  const auto connection_count = population.genome.num_connections[idx];
  if (node_count > moonai_gpu::kCompileScratchNodeLimit || connection_count > moonai_gpu::kCompileScratchConnectionLimit) {
    population.compiled.node_counts[idx] = 0U;
    population.compiled.eval_counts[idx] = 0U;
    population.compiled.connection_counts[idx] = 0U;
    return;
  }

  std::uint16_t incoming_counts[moonai_gpu::kCompileScratchNodeLimit]{};
  std::uint16_t write_positions[moonai_gpu::kCompileScratchNodeLimit]{};
  std::uint16_t indegree[moonai_gpu::kCompileScratchNodeLimit]{};
  std::uint16_t queue[moonai_gpu::kCompileScratchNodeLimit]{};

  const auto connection_base = static_cast<std::size_t>(idx) * population.genome.connection_stride;
  const auto offset_base = static_cast<std::size_t>(idx) * (population.compiled.node_stride + 1U);
  const auto compiled_connection_base = static_cast<std::size_t>(idx) * population.compiled.connection_stride;
  const auto eval_base = static_cast<std::size_t>(idx) * population.compiled.node_stride;
  const auto output_base = static_cast<std::size_t>(idx) * population.compiled.output_stride;

  std::uint16_t active_connection_count = 0U;
  for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
    const auto entry = connection_base + connection;
    if (population.genome.connection_enabled[entry] == 0U) {
      continue;
    }
    const auto from_node = population.genome.connection_from[entry];
    const auto to_node = population.genome.connection_to[entry];
    if (from_node < 0 || to_node < 0 || static_cast<std::uint16_t>(from_node) >= node_count ||
        static_cast<std::uint16_t>(to_node) >= node_count) {
      continue;
    }
    ++incoming_counts[static_cast<std::uint16_t>(to_node)];
    ++indegree[static_cast<std::uint16_t>(to_node)];
    ++active_connection_count;
  }

  std::uint32_t running_offset = 0U;
  for (std::uint16_t node = 0; node < node_count; ++node) {
    population.compiled.connection_offsets[offset_base + node] = running_offset;
    write_positions[node] = static_cast<std::uint16_t>(running_offset);
    running_offset += incoming_counts[node];
  }
  population.compiled.connection_offsets[offset_base + node_count] = running_offset;
  for (std::uint32_t node = node_count + 1U; node <= population.compiled.node_stride; ++node) {
    population.compiled.connection_offsets[offset_base + node] = running_offset;
  }

  for (std::uint32_t connection = 0; connection < population.compiled.connection_stride; ++connection) {
    population.compiled.connection_sources[compiled_connection_base + connection] = 0U;
    population.compiled.connection_weights[compiled_connection_base + connection] = 0.0F;
  }
  for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
    const auto entry = connection_base + connection;
    if (population.genome.connection_enabled[entry] == 0U) {
      continue;
    }
    const auto from_node = population.genome.connection_from[entry];
    const auto to_node = population.genome.connection_to[entry];
    if (from_node < 0 || to_node < 0 || static_cast<std::uint16_t>(from_node) >= node_count ||
        static_cast<std::uint16_t>(to_node) >= node_count) {
      continue;
    }
    const auto target = static_cast<std::uint16_t>(to_node);
    const auto write_index = write_positions[target]++;
    population.compiled.connection_sources[compiled_connection_base + write_index] = static_cast<std::uint16_t>(from_node);
    population.compiled.connection_weights[compiled_connection_base + write_index] = population.genome.connection_weight[entry];
  }

  std::uint16_t queue_head = 0U;
  std::uint16_t queue_tail = 0U;
  for (std::uint16_t node = 0; node < node_count; ++node) {
    if (indegree[node] == 0U) {
      queue[queue_tail++] = node;
    }
  }

  std::uint16_t visited_nodes = 0U;
  std::uint16_t eval_count = 0U;
  while (queue_head < queue_tail) {
    const auto node = queue[queue_head++];
    ++visited_nodes;

    const auto node_type = population.genome.node_types[static_cast<std::size_t>(idx) * population.genome.node_stride + node];
    if (node_type == moonai_gpu::kHiddenNodeType || node_type == moonai_gpu::kOutputNodeType) {
      population.compiled.eval_order[eval_base + eval_count] = node;
      ++eval_count;
    }

    for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
      const auto entry = connection_base + connection;
      if (population.genome.connection_enabled[entry] == 0U || population.genome.connection_from[entry] != node) {
        continue;
      }
      const auto to_node = population.genome.connection_to[entry];
      if (to_node < 0 || static_cast<std::uint16_t>(to_node) >= node_count) {
        continue;
      }
      auto &target_indegree = indegree[static_cast<std::uint16_t>(to_node)];
      if (target_indegree > 0U) {
        --target_indegree;
        if (target_indegree == 0U) {
          queue[queue_tail++] = static_cast<std::uint16_t>(to_node);
        }
      }
    }
  }

  if (visited_nodes != node_count) {
    eval_count = 0U;
    for (std::uint16_t node = 0; node < node_count; ++node) {
      const auto node_type = population.genome.node_types[static_cast<std::size_t>(idx) * population.genome.node_stride + node];
      if (node_type == moonai_gpu::kHiddenNodeType || node_type == moonai_gpu::kOutputNodeType) {
        population.compiled.eval_order[eval_base + eval_count] = node;
        ++eval_count;
      }
    }
  }
  for (std::uint32_t eval = eval_count; eval < population.compiled.node_stride; ++eval) {
    population.compiled.eval_order[eval_base + eval] = 0U;
  }

  std::uint16_t output_count = 0U;
  for (std::uint16_t node = 0; node < node_count; ++node) {
    const auto node_type = population.genome.node_types[static_cast<std::size_t>(idx) * population.genome.node_stride + node];
    if (node_type == moonai_gpu::kOutputNodeType && output_count < output_stride) {
      population.compiled.output_indices[output_base + output_count] = node;
      ++output_count;
    }
  }
  for (std::uint32_t output = output_count; output < population.compiled.output_stride; ++output) {
    population.compiled.output_indices[output_base + output] = 0U;
  }

  population.compiled.node_counts[idx] = node_count;
  population.compiled.eval_counts[idx] = eval_count;
  population.compiled.connection_counts[idx] = active_connection_count;
}

__global__ void compile_population_kernel(DevicePopulationBuffers population, std::uint32_t output_stride) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity) {
    return;
  }
  compile_slot_device(population, idx, output_stride);
}

__global__ void compile_single_slot_kernel(DevicePopulationBuffers population, std::uint32_t slot,
                                           std::uint32_t output_stride) {
  if (blockIdx.x != 0U || threadIdx.x != 0U || slot >= population.capacity) {
    return;
  }
  compile_slot_device(population, slot, output_stride);
}

__global__ void compile_slots_kernel(DevicePopulationBuffers population, const std::uint32_t *free_list,
                                     std::uint32_t free_slot_base, std::uint32_t births_applied,
                                     std::uint32_t output_stride) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= births_applied) {
    return;
  }

  compile_slot_device(population, free_list[free_slot_base + idx], output_stride);
}

__global__ void compiled_header_kernel(const DevicePopulationBuffers population, PopulationKind population_kind,
                                       std::uint32_t slot, std::uint32_t output_stride,
                                       CompiledNetworkReadbackHeader *out_header) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  out_header->population_kind = population_kind;
  out_header->slot = slot;
  out_header->node_count = population.compiled.node_counts[slot];
  out_header->eval_node_count = population.compiled.eval_counts[slot];
  out_header->output_count = static_cast<std::uint16_t>(output_stride);
  out_header->connection_count = population.compiled.connection_counts[slot];
}

__global__ void selected_agent_network_kernel(const DevicePopulationBuffers population, PopulationKind population_kind,
                                              std::uint32_t slot, std::uint32_t num_inputs,
                                              SelectedAgentNetworkReadback *out_network) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  float activations[moonai_gpu::kCompileScratchNodeLimit]{};
  const auto node_count = moonai_gpu::evaluate_compiled_network(population, slot, num_inputs, activations);

  out_network->population_kind = population_kind;
  out_network->slot = slot;
  out_network->node_count = node_count;
  out_network->output_count = static_cast<std::uint16_t>(population.compiled.output_stride);
  out_network->activation_count = node_count;
  out_network->reserved = 0U;
  out_network->output_0 = moonai_gpu::compiled_output_activation(population, slot, node_count, activations, 0U);
  out_network->output_1 = moonai_gpu::compiled_output_activation(population, slot, node_count, activations, 1U);
}

__global__ void reset_species_reduce_scratch_kernel(SpeciesReduceScratch *scratch, std::uint32_t *out_species_count) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < moonai_gpu::kSpeciesBucketCount) {
    scratch->sizes[idx] = 0U;
    scratch->complexity_sums[idx] = 0.0F;
    scratch->representative_slots[idx] = 0xFFFF'FFFFU;
  }
  if (idx == 0U) {
    *out_species_count = 0U;
  }
}

template <PopulationKind Kind>
__global__ void accumulate_species_kernel(DevicePopulationBuffers population, SpeciesReduceScratch *scratch) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  const auto node_count = population.genome.num_nodes[idx];
  const auto connection_count = population.genome.num_connections[idx];
  const auto enabled_count = moonai_gpu::count_enabled_connections(population, idx, connection_count);
  const auto species_id = species_bucket(population, idx, node_count, connection_count);
  const auto complexity = static_cast<float>(node_count + enabled_count);
  population.species_id[idx] = species_id;
  atomicAdd(&scratch->sizes[species_id], 1U);
  atomicAdd(&scratch->complexity_sums[species_id], complexity);
  atomicMin(&scratch->representative_slots[species_id], idx);
}

__global__ void write_species_summaries_kernel(DevicePopulationBuffers population, PopulationKind population_kind,
                                               const SpeciesReduceScratch *scratch,
                                               SpeciesSummaryReadback *out_summaries,
                                               RepresentativeGenomeHeader *out_headers,
                                               std::uint32_t *out_species_count) {
  const auto species_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (species_id >= moonai_gpu::kSpeciesBucketCount) {
    return;
  }

  const auto size = scratch->sizes[species_id];
  if (size == 0U) {
    return;
  }

  const auto dense_index = atomicAdd(out_species_count, 1U);
  const auto representative_slot = scratch->representative_slots[species_id];
  out_summaries[dense_index] = SpeciesSummaryReadback{population_kind, species_id, size, representative_slot,
                                                      scratch->complexity_sums[species_id] /
                                                          static_cast<float>(size)};
  build_representative_header(population, population_kind, representative_slot, &out_headers[dense_index]);
}

} // namespace

extern "C" std::int32_t dev_compile_population(DeviceState *state, PopulationKind population_kind) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);

  const auto blocks = (population.capacity + 255U) / 256U;
  compile_population_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, state->num_outputs);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_compile_slot(DeviceState *state, PopulationKind population_kind, std::uint32_t slot) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  compile_single_slot_kernel<<<1U, 1U>>>(population, slot, state->num_outputs);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_compile_slots(DeviceState *state, PopulationKind population_kind,
                                           std::uint32_t births_applied, std::uint32_t free_slot_base) {
  if (births_applied == 0U) {
    return 0;
  }

  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *free_list = population_kind == PopulationKind::Predator ? state->predator_free_list : state->prey_free_list;
  const auto blocks = (births_applied + 255U) / 256U;
  compile_slots_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, free_list, free_slot_base, births_applied,
                                                              state->num_outputs);
  return moonai_gpu::launch_status();
}

extern "C" std::int32_t dev_write_compiled_header(const DeviceState *state, PopulationKind population_kind, std::uint32_t slot) {
  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  compiled_header_kernel<<<1U, 1U>>>(population, population_kind, slot, state->num_outputs, state->compiled_header_scratch);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_write_selected_agent_network(const DeviceState *state, PopulationKind population_kind, std::uint32_t slot) {
  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  selected_agent_network_kernel<<<1U, 1U>>>(population, population_kind, slot, state->num_inputs, state->selected_network_scratch);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_classify_species_summaries(DeviceState *state, PopulationKind population_kind) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  reset_species_reduce_scratch_kernel<<<1U, moonai_gpu::kSpeciesBucketCount>>>(state->species_reduce_scratch,
                                                                                state->species_count_scratch);
  const auto blocks = (population.capacity + 255U) / 256U;
  if (population_kind == PopulationKind::Predator) {
    accumulate_species_kernel<PopulationKind::Predator><<<blocks == 0U ? 1U : blocks, 256U>>>(population,
                                                                                                 state->species_reduce_scratch);
  } else {
    accumulate_species_kernel<PopulationKind::Prey><<<blocks == 0U ? 1U : blocks, 256U>>>(population,
                                                                                             state->species_reduce_scratch);
  }
  write_species_summaries_kernel<<<1U, moonai_gpu::kSpeciesBucketCount>>>(population, population_kind,
                                                                           state->species_reduce_scratch,
                                                                           state->species_summaries_scratch,
                                                                           state->representative_headers_scratch,
                                                                           state->species_count_scratch);
  return moonai_gpu::launch_status();
}
