#include "evolution_cuda.cuh"

using moonai_gpu::CompiledNetworkReadbackHeader;
using moonai_gpu::CudaStatus;
using moonai_gpu::DeviceInnovationState;
using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::GpuEvolutionState;
using moonai_gpu::InvariantCheckReadback;
using moonai_gpu::PopulationKind;
using moonai_gpu::RepresentativeGenomeHeader;
using moonai_gpu::SelectedAgentNetworkReadback;
using moonai_gpu::SpeciesBatchReadbackHeader;
using moonai_gpu::SpeciesSummaryReadback;

namespace {

__device__ std::uint16_t count_enabled_connections(const DevicePopulationBuffers &population, std::uint32_t slot,
                                                   std::uint16_t connection_count) {
  const auto connection_base = static_cast<std::size_t>(slot) * population.genome.connection_stride;
  std::uint16_t enabled_count = 0U;
  for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
    enabled_count = static_cast<std::uint16_t>(enabled_count + (population.genome.connection_enabled[connection_base + connection] != 0U));
  }
  return enabled_count;
}

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
  const auto enabled_count = count_enabled_connections(population, slot, connection_count);
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
  const auto node_count = population.compiled.node_counts[slot];
  const auto eval_count = population.compiled.eval_counts[slot];
  const auto offset_base = static_cast<std::size_t>(slot) * (population.compiled.node_stride + 1U);
  const auto eval_base = static_cast<std::size_t>(slot) * population.compiled.node_stride;
  const auto output_base = static_cast<std::size_t>(slot) * population.compiled.output_stride;
  const auto compiled_connection_base = static_cast<std::size_t>(slot) * population.compiled.connection_stride;

  if (node_count > 0U && num_inputs < node_count) {
    activations[num_inputs] = 1.0F;
  }
  for (std::uint16_t eval_index = 0; eval_index < eval_count; ++eval_index) {
    const auto node = population.compiled.eval_order[eval_base + eval_index];
    if (node >= node_count) {
      continue;
    }
    const auto start = population.compiled.connection_offsets[offset_base + node];
    const auto end = population.compiled.connection_offsets[offset_base + node + 1U];
    float sum = 0.0F;
    for (std::uint32_t connection = start; connection < end; ++connection) {
      const auto source = population.compiled.connection_sources[compiled_connection_base + connection];
      if (source < node_count) {
        sum += activations[source] * population.compiled.connection_weights[compiled_connection_base + connection];
      }
    }
    activations[node] = tanhf(sum);
  }

  out_network->population_kind = population_kind;
  out_network->slot = slot;
  out_network->node_count = node_count;
  out_network->output_count = static_cast<std::uint16_t>(population.compiled.output_stride);
  out_network->activation_count = node_count;
  out_network->reserved = 0U;
  out_network->output_0 =
      population.compiled.output_stride > 0U && population.compiled.output_indices[output_base] < node_count
          ? activations[population.compiled.output_indices[output_base]]
          : 0.0F;
  out_network->output_1 =
      population.compiled.output_stride > 1U && population.compiled.output_indices[output_base + 1U] < node_count
          ? activations[population.compiled.output_indices[output_base + 1U]]
          : 0.0F;
}

__global__ void classify_species_kernel(DevicePopulationBuffers population, PopulationKind population_kind,
                                        std::uint32_t slot, SpeciesSummaryReadback *out_summary,
                                        RepresentativeGenomeHeader *out_header) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  std::uint32_t target_species = 0U;
  bool target_alive = false;
  for (std::uint32_t idx = 0; idx < population.capacity; ++idx) {
    if (population.alive[idx] == 0U) {
      continue;
    }
    const auto node_count = population.genome.num_nodes[idx];
    const auto connection_count = population.genome.num_connections[idx];
    const auto species_id = species_bucket(population, idx, node_count, connection_count);
    population.species_id[idx] = species_id;
    if (idx == slot) {
      target_species = species_id;
      target_alive = true;
    }
  }

  out_summary->population_kind = population_kind;
  out_summary->species_id = target_species;
  out_summary->size = 0U;
  out_summary->representative_slot = slot;
  out_summary->avg_complexity = 0.0F;
  *out_header = RepresentativeGenomeHeader{population_kind, slot, 0U, 0U, target_species, 0U, 0U};
  if (!target_alive) {
    return;
  }

  std::uint32_t representative_slot = slot;
  float complexity_sum = 0.0F;
  bool representative_set = false;
  for (std::uint32_t idx = 0; idx < population.capacity; ++idx) {
    if (population.alive[idx] == 0U || population.species_id[idx] != target_species) {
      continue;
    }
    const auto node_count = population.genome.num_nodes[idx];
    const auto connection_count = population.genome.num_connections[idx];
    complexity_sum += static_cast<float>(node_count + count_enabled_connections(population, idx, connection_count));
    ++out_summary->size;
    if (!representative_set) {
      representative_slot = idx;
      representative_set = true;
    }
  }

  out_summary->representative_slot = representative_slot;
  out_summary->avg_complexity = out_summary->size == 0U ? 0.0F : complexity_sum / static_cast<float>(out_summary->size);
  build_representative_header(population, population_kind, representative_slot, out_header);
}

__global__ void classify_species_batch_kernel(DevicePopulationBuffers population, PopulationKind population_kind,
                                              SpeciesSummaryReadback *out_summaries,
                                              RepresentativeGenomeHeader *out_headers,
                                              std::uint32_t *out_species_count) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  std::uint32_t species_sizes[moonai_gpu::kSpeciesBucketCount]{};
  float complexity_sums[moonai_gpu::kSpeciesBucketCount]{};
  std::uint32_t representative_slots[moonai_gpu::kSpeciesBucketCount]{};
  bool representative_seen[moonai_gpu::kSpeciesBucketCount]{};

  for (std::uint32_t idx = 0; idx < population.capacity; ++idx) {
    if (population.alive[idx] == 0U) {
      continue;
    }

    const auto node_count = population.genome.num_nodes[idx];
    const auto connection_count = population.genome.num_connections[idx];
    const auto enabled_count = count_enabled_connections(population, idx, connection_count);
    const auto species_id = species_bucket(population, idx, node_count, connection_count);
    population.species_id[idx] = species_id;

    if (!representative_seen[species_id]) {
      representative_slots[species_id] = idx;
      representative_seen[species_id] = true;
    }

    ++species_sizes[species_id];
    complexity_sums[species_id] += static_cast<float>(node_count + enabled_count);
  }

  std::uint32_t dense_count = 0U;
  for (std::uint32_t species_id = 0; species_id < moonai_gpu::kSpeciesBucketCount; ++species_id) {
    if (species_sizes[species_id] == 0U) {
      continue;
    }

    const auto representative_slot = representative_slots[species_id];
    out_summaries[dense_count] = SpeciesSummaryReadback{population_kind,
                                                        species_id,
                                                        species_sizes[species_id],
                                                        representative_slot,
                                                        complexity_sums[species_id] /
                                                            static_cast<float>(species_sizes[species_id])};
    build_representative_header(population, population_kind, representative_slot, &out_headers[dense_count]);
    ++dense_count;
  }

  *out_species_count = dense_count;
}

__device__ void inspect_population_invariants(const DevicePopulationBuffers &population, std::uint32_t output_stride,
                                              std::uint32_t *agents_checked, InvariantCheckReadback *out_checks) {
  for (std::uint32_t slot = 0; slot < population.capacity; ++slot) {
    if (population.alive[slot] == 0U) {
      continue;
    }
    ++(*agents_checked);

    const auto node_count = population.genome.num_nodes[slot];
    const auto connection_count = population.genome.num_connections[slot];
    if (node_count == 0U || node_count > population.genome.node_stride) {
      ++out_checks->invalid_node_counts;
      continue;
    }
    if (connection_count > population.genome.connection_stride) {
      ++out_checks->invalid_connection_counts;
      continue;
    }

    const auto connection_base = static_cast<std::size_t>(slot) * population.genome.connection_stride;
    for (std::uint16_t connection = 0; connection < connection_count; ++connection) {
      const auto entry = connection_base + connection;
      const auto from_node = population.genome.connection_from[entry];
      const auto to_node = population.genome.connection_to[entry];
      if (from_node < 0 || to_node < 0 || static_cast<std::uint16_t>(from_node) >= node_count ||
          static_cast<std::uint16_t>(to_node) >= node_count) {
        ++out_checks->invalid_connection_bounds;
      }
    }

    const auto compiled_node_count = population.compiled.node_counts[slot];
    const auto compiled_eval_count = population.compiled.eval_counts[slot];
    const auto compiled_connection_count = population.compiled.connection_counts[slot];
    if (compiled_node_count != node_count || compiled_connection_count > population.compiled.connection_stride) {
      ++out_checks->invalid_compiled_offsets;
    }
    if (compiled_eval_count > compiled_node_count) {
      ++out_checks->invalid_eval_nodes;
    }

    const auto offset_base = static_cast<std::size_t>(slot) * (population.compiled.node_stride + 1U);
    std::uint32_t last_offset = 0U;
    for (std::uint16_t node = 0; node < compiled_node_count + 1U; ++node) {
      const auto offset = population.compiled.connection_offsets[offset_base + node];
      if (offset < last_offset || offset > compiled_connection_count) {
        ++out_checks->invalid_compiled_offsets;
        break;
      }
      last_offset = offset;
    }

    const auto eval_base = static_cast<std::size_t>(slot) * population.compiled.node_stride;
    for (std::uint16_t eval = 0; eval < compiled_eval_count; ++eval) {
      if (population.compiled.eval_order[eval_base + eval] >= compiled_node_count) {
        ++out_checks->invalid_eval_nodes;
      }
    }

    const auto output_base = static_cast<std::size_t>(slot) * population.compiled.output_stride;
    for (std::uint32_t output = 0; output < output_stride; ++output) {
      if (population.compiled.output_indices[output_base + output] >= compiled_node_count) {
        ++out_checks->invalid_output_indices;
      }
    }

    if (population.species_id[slot] >= moonai_gpu::kSpeciesBucketCount) {
      ++out_checks->invalid_species_assignments;
    }
  }
}

__global__ void invariant_check_kernel(const DevicePopulationBuffers predator, const DevicePopulationBuffers prey,
                                       const DeviceInnovationState *innovation, std::uint32_t output_stride,
                                       InvariantCheckReadback *out_checks) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  *out_checks = InvariantCheckReadback{};
  inspect_population_invariants(predator, output_stride, &out_checks->predator_agents_checked, out_checks);
  inspect_population_invariants(prey, output_stride, &out_checks->prey_agents_checked, out_checks);
  out_checks->innovation_log_overflow =
      innovation->log_len > innovation->log_capacity ? innovation->log_len - innovation->log_capacity : 0U;
}

} // namespace

extern "C" std::int32_t moonai_gpu_evolution_compile_population(void *state_ptr, PopulationKind population_kind,
                                                                   std::uint32_t inspected_slot,
                                                                   CompiledNetworkReadbackHeader *out_header) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_header == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  if (inspected_slot >= population.capacity) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  CompiledNetworkReadbackHeader *device_header = nullptr;
  auto status = moonai_gpu::alloc_array(&device_header, 1U);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  const auto blocks = (population.capacity + 255U) / 256U;
  compile_population_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, state->config.num_outputs);
  status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    compiled_header_kernel<<<1U, 1U>>>(population, population_kind, inspected_slot, state->config.num_outputs, device_header);
    status = moonai_gpu::synchronize_kernels();
  }
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_header, out_header, sizeof(*out_header));
  }

  moonai_gpu::free_array(device_header);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_compile_slot(void *state_ptr, PopulationKind population_kind,
                                                             std::uint32_t slot,
                                                             CompiledNetworkReadbackHeader *out_header) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_header == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  if (slot >= population.capacity) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  CompiledNetworkReadbackHeader *device_header = nullptr;
  auto status = moonai_gpu::alloc_array(&device_header, 1U);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  compile_single_slot_kernel<<<1U, 1U>>>(population, slot, state->config.num_outputs);
  status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    compiled_header_kernel<<<1U, 1U>>>(population, population_kind, slot, state->config.num_outputs, device_header);
    status = moonai_gpu::synchronize_kernels();
  }
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_header, out_header, sizeof(*out_header));
  }

  moonai_gpu::free_array(device_header);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_selected_agent_network(const void *state_ptr,
                                                                       PopulationKind population_kind,
                                                                       std::uint32_t slot,
                                                                      SelectedAgentNetworkReadback *out_network) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_network == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  if (slot >= population.capacity) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  SelectedAgentNetworkReadback *device_network = nullptr;
  auto status = moonai_gpu::alloc_array(&device_network, 1U);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  selected_agent_network_kernel<<<1U, 1U>>>(population, population_kind, slot, state->config.num_inputs, device_network);
  status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_network, out_network, sizeof(*out_network));
  }

  moonai_gpu::free_array(device_network);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_classify_species(void *state_ptr, PopulationKind population_kind,
                                                                std::uint32_t slot,
                                                                SpeciesSummaryReadback *out_summary,
                                                                RepresentativeGenomeHeader *out_header) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_summary == nullptr || out_header == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  if (slot >= population.capacity) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  SpeciesSummaryReadback *device_summary = nullptr;
  RepresentativeGenomeHeader *device_header = nullptr;
  auto status = moonai_gpu::alloc_array(&device_summary, 1U);
  if (status == CudaStatus::Success) {
    status = moonai_gpu::alloc_array(&device_header, 1U);
  }
  if (status == CudaStatus::Success) {
    classify_species_kernel<<<1U, 1U>>>(population, population_kind, slot, device_summary, device_header);
    status = moonai_gpu::synchronize_kernels();
  }
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_summary, out_summary, sizeof(*out_summary));
  }
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_header, out_header, sizeof(*out_header));
  }

  moonai_gpu::free_array(device_summary);
  moonai_gpu::free_array(device_header);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_species_summaries(void *state_ptr, PopulationKind population_kind,
                                                                 std::uint32_t max_species,
                                                                 SpeciesBatchReadbackHeader *out_header,
                                                                 SpeciesSummaryReadback *out_summaries,
                                                                 RepresentativeGenomeHeader *out_representatives) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_header == nullptr ||
      (max_species != 0U && (out_summaries == nullptr || out_representatives == nullptr))) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  SpeciesSummaryReadback *device_summaries = nullptr;
  RepresentativeGenomeHeader *device_headers = nullptr;
  std::uint32_t *device_count = nullptr;
  auto status = moonai_gpu::alloc_array(&device_summaries, moonai_gpu::kSpeciesBucketCount);
  if (status == CudaStatus::Success) {
    status = moonai_gpu::alloc_array(&device_headers, moonai_gpu::kSpeciesBucketCount);
  }
  if (status == CudaStatus::Success) {
    status = moonai_gpu::alloc_array(&device_count, 1U);
  }
  if (status == CudaStatus::Success) {
    classify_species_batch_kernel<<<1U, 1U>>>(population, population_kind, device_summaries, device_headers, device_count);
    status = moonai_gpu::synchronize_kernels();
  }

  std::uint32_t species_count = 0U;
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_count, &species_count, sizeof(species_count));
  }
  if (status == CudaStatus::Success) {
    const auto returned_species_count = species_count < max_species ? species_count : max_species;
    out_header->population_kind = population_kind;
    out_header->species_count = species_count;
    out_header->returned_species_count = returned_species_count;
    if (returned_species_count > 0U) {
      status = moonai_gpu::copy_compact_device_readback(device_summaries, out_summaries,
                                                        sizeof(SpeciesSummaryReadback) * returned_species_count);
    }
    if (status == CudaStatus::Success && returned_species_count > 0U) {
      status = moonai_gpu::copy_compact_device_readback(device_headers, out_representatives,
                                                        sizeof(RepresentativeGenomeHeader) * returned_species_count);
    }
  }

  moonai_gpu::free_array(device_summaries);
  moonai_gpu::free_array(device_headers);
  moonai_gpu::free_array(device_count);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_check_invariants(const void *state_ptr,
                                                                InvariantCheckReadback *out_checks) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_checks == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  InvariantCheckReadback *device_checks = nullptr;
  auto status = moonai_gpu::alloc_array(&device_checks, 1U);
  if (status == CudaStatus::Success) {
    invariant_check_kernel<<<1U, 1U>>>(state->predator, state->prey, state->innovation, state->config.num_outputs, device_checks);
    status = moonai_gpu::synchronize_kernels();
  }
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_checks, out_checks, sizeof(*out_checks));
  }

  moonai_gpu::free_array(device_checks);
  return static_cast<std::int32_t>(status);
}
