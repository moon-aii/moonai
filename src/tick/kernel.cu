#include "evolution_cuda.cuh"

#include <new>

using moonai_gpu::CudaStatus;
using moonai_gpu::DeviceInnovationState;
using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::GpuEvolutionConfig;
using moonai_gpu::GpuEvolutionState;
using moonai_gpu::InnovationRecord;
using moonai_gpu::PopulationKind;
using moonai_gpu::PopulationSummaryReadback;
using moonai_gpu::SeededAgentSnapshot;
using moonai_gpu::UiStatsReadback;

namespace moonai_gpu {

std::int32_t g_last_cuda_error_code = 0;

} // namespace moonai_gpu

namespace {

CudaStatus allocate_population_buffers(DevicePopulationBuffers &population, std::uint32_t capacity,
                                       std::uint32_t node_stride, std::uint32_t connection_stride,
                                       std::uint32_t output_stride) {
  population.capacity = capacity;
  population.genome.connection_stride = connection_stride;
  population.genome.node_stride = node_stride;
  population.compiled.node_stride = node_stride;
  population.compiled.connection_stride = connection_stride;
  population.compiled.output_stride = output_stride;

  for (auto status : {
           moonai_gpu::alloc_array(&population.pos_x, capacity),
           moonai_gpu::alloc_array(&population.pos_y, capacity),
           moonai_gpu::alloc_array(&population.vel_x, capacity),
           moonai_gpu::alloc_array(&population.vel_y, capacity),
           moonai_gpu::alloc_array(&population.energy, capacity),
           moonai_gpu::alloc_array(&population.age, capacity),
           moonai_gpu::alloc_array(&population.alive, capacity),
           moonai_gpu::alloc_array(&population.species_id, capacity),
           moonai_gpu::alloc_array(&population.entity_id, capacity),
           moonai_gpu::alloc_array(&population.generation, capacity),
           moonai_gpu::alloc_array(&population.rng_state, capacity),
           moonai_gpu::alloc_array(&population.genome.connection_from, static_cast<std::size_t>(capacity) * connection_stride),
           moonai_gpu::alloc_array(&population.genome.connection_to, static_cast<std::size_t>(capacity) * connection_stride),
           moonai_gpu::alloc_array(&population.genome.connection_weight, static_cast<std::size_t>(capacity) * connection_stride),
           moonai_gpu::alloc_array(&population.genome.connection_innovation,
                                   static_cast<std::size_t>(capacity) * connection_stride),
           moonai_gpu::alloc_array(&population.genome.connection_enabled,
                                   static_cast<std::size_t>(capacity) * connection_stride),
           moonai_gpu::alloc_array(&population.genome.node_types, static_cast<std::size_t>(capacity) * node_stride),
           moonai_gpu::alloc_array(&population.genome.num_connections, capacity),
           moonai_gpu::alloc_array(&population.genome.num_nodes, capacity),
           moonai_gpu::alloc_array(&population.compiled.eval_order, static_cast<std::size_t>(capacity) * node_stride),
           moonai_gpu::alloc_array(&population.compiled.connection_offsets,
                                   static_cast<std::size_t>(capacity) * (node_stride + 1U)),
           moonai_gpu::alloc_array(&population.compiled.output_indices, static_cast<std::size_t>(capacity) * output_stride),
           moonai_gpu::alloc_array(&population.compiled.connection_sources,
                                   static_cast<std::size_t>(capacity) * connection_stride),
           moonai_gpu::alloc_array(&population.compiled.connection_weights,
                                   static_cast<std::size_t>(capacity) * connection_stride),
           moonai_gpu::alloc_array(&population.compiled.node_counts, capacity),
           moonai_gpu::alloc_array(&population.compiled.eval_counts, capacity),
           moonai_gpu::alloc_array(&population.compiled.connection_counts, capacity),
       }) {
    if (status != CudaStatus::Success) {
      return status;
    }
  }

  return CudaStatus::Success;
}

void free_population_buffers(DevicePopulationBuffers &population) {
  moonai_gpu::free_array(population.pos_x);
  moonai_gpu::free_array(population.pos_y);
  moonai_gpu::free_array(population.vel_x);
  moonai_gpu::free_array(population.vel_y);
  moonai_gpu::free_array(population.energy);
  moonai_gpu::free_array(population.age);
  moonai_gpu::free_array(population.alive);
  moonai_gpu::free_array(population.species_id);
  moonai_gpu::free_array(population.entity_id);
  moonai_gpu::free_array(population.generation);
  moonai_gpu::free_array(population.rng_state);
  moonai_gpu::free_array(population.genome.connection_from);
  moonai_gpu::free_array(population.genome.connection_to);
  moonai_gpu::free_array(population.genome.connection_weight);
  moonai_gpu::free_array(population.genome.connection_innovation);
  moonai_gpu::free_array(population.genome.connection_enabled);
  moonai_gpu::free_array(population.genome.node_types);
  moonai_gpu::free_array(population.genome.num_connections);
  moonai_gpu::free_array(population.genome.num_nodes);
  moonai_gpu::free_array(population.compiled.eval_order);
  moonai_gpu::free_array(population.compiled.connection_offsets);
  moonai_gpu::free_array(population.compiled.output_indices);
  moonai_gpu::free_array(population.compiled.connection_sources);
  moonai_gpu::free_array(population.compiled.connection_weights);
  moonai_gpu::free_array(population.compiled.node_counts);
  moonai_gpu::free_array(population.compiled.eval_counts);
  moonai_gpu::free_array(population.compiled.connection_counts);
  population.capacity = 0U;
}

void destroy_state(GpuEvolutionState *state) {
  if (state == nullptr) {
    return;
  }
  free_population_buffers(state->predator);
  free_population_buffers(state->prey);
  moonai_gpu::free_array(state->innovation);
  moonai_gpu::free_array(state->innovation_log);
  moonai_gpu::free_array(state->next_entity_id);
  delete state;
}

__global__ void seed_population_kernel(DevicePopulationBuffers population, std::uint32_t live_count,
                                       std::uint32_t base_entity_id, std::uint64_t base_seed,
                                       std::uint32_t num_inputs, std::uint32_t num_outputs, float world_size,
                                       float initial_energy, float max_energy, PopulationKind population_kind) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity) {
    return;
  }

  const auto alive = idx < live_count ? 1U : 0U;
  std::uint64_t rng = base_seed ^
                      (static_cast<std::uint64_t>(population_kind == PopulationKind::Predator ? 0xA51U : 0xB29U) << 32U) ^
                      idx;
  rng = moonai_gpu::splitmix64(rng);

  population.pos_x[idx] = alive == 0U ? 0.0F : moonai_gpu::next_unit_float(rng) * world_size;
  population.pos_y[idx] = alive == 0U ? 0.0F : moonai_gpu::next_unit_float(rng) * world_size;
  population.vel_x[idx] = 0.0F;
  population.vel_y[idx] = 0.0F;
  population.energy[idx] = alive == 0U ? 0.0F : (initial_energy < max_energy ? initial_energy : max_energy);
  population.age[idx] = 0.0F;
  population.alive[idx] = static_cast<std::uint8_t>(alive);
  population.species_id[idx] = 0U;
  population.entity_id[idx] = alive == 0U ? 0U : base_entity_id + idx;
  population.generation[idx] = 0U;
  population.rng_state[idx] = rng;

  const auto seeded_node_count = static_cast<std::uint16_t>(num_inputs + num_outputs + 1U);
  const auto seeded_connection_count = static_cast<std::uint16_t>((num_inputs + 1U) * num_outputs);
  population.genome.num_nodes[idx] = alive == 0U ? 0U : seeded_node_count;
  population.genome.num_connections[idx] = alive == 0U ? 0U : seeded_connection_count;

  const auto node_base = static_cast<std::size_t>(idx) * population.genome.node_stride;
  for (std::uint32_t node = 0; node < population.genome.node_stride; ++node) {
    population.genome.node_types[node_base + node] = moonai_gpu::kInputNodeType;
  }
  if (alive != 0U) {
    for (std::uint32_t node = 0; node < num_inputs; ++node) {
      population.genome.node_types[node_base + node] = moonai_gpu::kInputNodeType;
    }
    population.genome.node_types[node_base + num_inputs] = moonai_gpu::kBiasNodeType;
    for (std::uint32_t node = 0; node < num_outputs; ++node) {
      population.genome.node_types[node_base + num_inputs + 1U + node] = moonai_gpu::kOutputNodeType;
    }
  }

  const auto connection_base = static_cast<std::size_t>(idx) * population.genome.connection_stride;
  for (std::uint32_t connection = 0; connection < population.genome.connection_stride; ++connection) {
    population.genome.connection_from[connection_base + connection] = 0;
    population.genome.connection_to[connection_base + connection] = 0;
    population.genome.connection_weight[connection_base + connection] = 0.0F;
    population.genome.connection_innovation[connection_base + connection] = 0U;
    population.genome.connection_enabled[connection_base + connection] = 0U;
  }
  if (alive != 0U) {
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
  population.compiled.node_counts[idx] = alive == 0U ? 0U : seeded_node_count;
  population.compiled.eval_counts[idx] = 0U;
  population.compiled.connection_counts[idx] = 0U;
}

__global__ void summarize_population_kernel(DevicePopulationBuffers population, const DeviceInnovationState *innovation,
                                            const std::uint32_t *next_entity_id, PopulationKind population_kind,
                                            PopulationSummaryReadback *out_summary) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  float total_energy = 0.0F;
  float total_connections = 0.0F;
  std::uint32_t live_count = 0U;
  for (std::uint32_t idx = 0; idx < population.capacity; ++idx) {
    if (population.alive[idx] == 0U) {
      continue;
    }
    ++live_count;
    total_energy += population.energy[idx];
    total_connections += static_cast<float>(population.genome.num_connections[idx]);
  }

  out_summary->population_kind = population_kind;
  out_summary->live_count = live_count;
  out_summary->capacity = population.capacity;
  out_summary->next_entity_id = *next_entity_id;
  out_summary->innovation_counter = innovation->next_innovation;
  out_summary->next_node_id = innovation->next_node_id;
  out_summary->avg_energy = live_count == 0U ? 0.0F : total_energy / static_cast<float>(live_count);
  out_summary->avg_connections = live_count == 0U ? 0.0F : total_connections / static_cast<float>(live_count);
}

__global__ void seeded_agent_snapshot_kernel(DevicePopulationBuffers population, PopulationKind population_kind,
                                             std::uint32_t slot, SeededAgentSnapshot *out_snapshot) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  out_snapshot->population_kind = population_kind;
  out_snapshot->slot = slot;
  out_snapshot->entity_id = population.entity_id[slot];
  out_snapshot->generation = population.generation[slot];
  out_snapshot->species_id = population.species_id[slot];
  out_snapshot->alive = population.alive[slot];
  out_snapshot->reserved0 = 0U;
  out_snapshot->reserved1 = 0U;
  out_snapshot->pos_x = population.pos_x[slot];
  out_snapshot->pos_y = population.pos_y[slot];
  out_snapshot->vel_x = population.vel_x[slot];
  out_snapshot->vel_y = population.vel_y[slot];
  out_snapshot->energy = population.energy[slot];
  out_snapshot->age = population.age[slot];
  out_snapshot->num_nodes = population.genome.num_nodes[slot];
  out_snapshot->num_connections = population.genome.num_connections[slot];

  std::uint64_t genome_hash = 1469598103934665603ULL;
  const auto node_base = static_cast<std::size_t>(slot) * population.genome.node_stride;
  for (std::uint32_t idx = 0; idx < population.genome.num_nodes[slot]; ++idx) {
    genome_hash = moonai_gpu::hash_mix(genome_hash, population.genome.node_types[node_base + idx]);
  }

  const auto connection_base = static_cast<std::size_t>(slot) * population.genome.connection_stride;
  for (std::uint32_t idx = 0; idx < population.genome.num_connections[slot]; ++idx) {
    const auto entry = connection_base + idx;
    genome_hash = moonai_gpu::hash_mix(genome_hash, static_cast<std::uint32_t>(population.genome.connection_from[entry]));
    genome_hash = moonai_gpu::hash_mix(genome_hash, static_cast<std::uint32_t>(population.genome.connection_to[entry]));
    genome_hash = moonai_gpu::hash_mix(genome_hash, population.genome.connection_innovation[entry]);
    genome_hash = moonai_gpu::hash_mix(genome_hash, population.genome.connection_enabled[entry]);
    genome_hash =
        moonai_gpu::hash_mix(genome_hash, static_cast<std::uint64_t>(__float_as_uint(population.genome.connection_weight[entry])));
  }

  out_snapshot->genome_hash = genome_hash;
}

} // namespace

extern "C" std::int32_t moonai_gpu_runtime_available() {
  const auto error = cudaFree(nullptr);
  const auto status = moonai_gpu::map_cuda_runtime_error(error, CudaStatus::KernelLaunchFailed);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_create(const GpuEvolutionConfig *config, void **out_state) {
  if (config == nullptr || out_state == nullptr || config->num_outputs == 0U || config->node_stride == 0U ||
      config->connection_stride == 0U || config->node_stride < config->num_inputs + config->num_outputs + 1U ||
      config->connection_stride < (config->num_inputs + 1U) * config->num_outputs) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  auto *state = new (std::nothrow) GpuEvolutionState{};
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::AllocationFailed);
  }
  state->config = *config;

  const auto total_capacity = static_cast<std::size_t>(config->predator_capacity) + config->prey_capacity;
  const auto innovation_log_capacity = total_capacity == 0U ? 1U : static_cast<std::uint32_t>(total_capacity * 4U);

  for (auto status : {
           allocate_population_buffers(state->predator, config->predator_capacity, config->node_stride,
                                       config->connection_stride, config->num_outputs),
           allocate_population_buffers(state->prey, config->prey_capacity, config->node_stride, config->connection_stride,
                                       config->num_outputs),
           moonai_gpu::alloc_array(&state->innovation, 1U), moonai_gpu::alloc_array(&state->innovation_log, innovation_log_capacity),
           moonai_gpu::alloc_array(&state->next_entity_id, 1U),
       }) {
    if (status != CudaStatus::Success) {
      destroy_state(state);
      return static_cast<std::int32_t>(status);
    }
  }

  *out_state = state;
  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" void moonai_gpu_evolution_destroy(void *state) { destroy_state(static_cast<GpuEvolutionState *>(state)); }

extern "C" std::int32_t moonai_gpu_evolution_seed_initial_population(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  seed_population_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, state->config.initial_predator_count, 1U, state->config.seed, state->config.num_inputs,
      state->config.num_outputs, state->config.world_size, state->config.initial_energy, state->config.max_energy,
      PopulationKind::Predator);
  seed_population_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, state->config.initial_prey_count, state->config.initial_predator_count + 1U,
      state->config.seed ^ 0x9e3779b97f4a7c15ULL, state->config.num_inputs, state->config.num_outputs,
      state->config.world_size, state->config.initial_energy, state->config.max_energy, PopulationKind::Prey);
  auto status = moonai_gpu::synchronize_kernels();
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  const auto innovation_log_capacity =
      (static_cast<std::size_t>(state->config.predator_capacity) + state->config.prey_capacity) == 0U
          ? 1U
          : static_cast<std::uint32_t>((static_cast<std::size_t>(state->config.predator_capacity) +
                                        state->config.prey_capacity) *
                                       4U);
  const DeviceInnovationState innovation_state{
      (state->config.num_inputs + 1U) * state->config.num_outputs,
      state->config.num_inputs + state->config.num_outputs + 1U,
      innovation_log_capacity,
      0U,
  };
  status = moonai_gpu::copy_host_data_to_device(state->innovation, &innovation_state, sizeof(innovation_state));
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  status = moonai_gpu::zero_device_memory(state->innovation_log, sizeof(InnovationRecord) * innovation_log_capacity);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  const std::uint32_t next_entity_id = state->config.initial_predator_count + state->config.initial_prey_count + 1U;
  status = moonai_gpu::copy_host_data_to_device(state->next_entity_id, &next_entity_id, sizeof(next_entity_id));
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" std::int32_t moonai_gpu_evolution_population_summary(const void *state_ptr, PopulationKind population_kind,
                                                                  PopulationSummaryReadback *out_summary) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_summary == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  PopulationSummaryReadback *device_summary = nullptr;
  auto status = moonai_gpu::alloc_array(&device_summary, 1U);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  summarize_population_kernel<<<1U, 1U>>>(moonai_gpu::population_for_kind(*state, population_kind), state->innovation,
                                          state->next_entity_id, population_kind, device_summary);
  status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_summary, out_summary, sizeof(*out_summary));
  }
  moonai_gpu::free_array(device_summary);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_seeded_agent_snapshot(const void *state_ptr, PopulationKind population_kind,
                                                                     std::uint32_t slot,
                                                                     SeededAgentSnapshot *out_snapshot) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_snapshot == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  if (slot >= population.capacity) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  SeededAgentSnapshot *device_snapshot = nullptr;
  auto status = moonai_gpu::alloc_array(&device_snapshot, 1U);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  seeded_agent_snapshot_kernel<<<1U, 1U>>>(population, population_kind, slot, device_snapshot);
  status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_snapshot, out_snapshot, sizeof(*out_snapshot));
  }
  moonai_gpu::free_array(device_snapshot);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_ui_stats(const void *state_ptr, UiStatsReadback *out_stats) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_stats == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  PopulationSummaryReadback predator_summary{};
  PopulationSummaryReadback prey_summary{};
  auto status = static_cast<CudaStatus>(
      moonai_gpu_evolution_population_summary(state_ptr, PopulationKind::Predator, &predator_summary));
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  status = static_cast<CudaStatus>(moonai_gpu_evolution_population_summary(state_ptr, PopulationKind::Prey, &prey_summary));
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  out_stats->tick = 0U;
  out_stats->predator_count = predator_summary.live_count;
  out_stats->prey_count = prey_summary.live_count;
  out_stats->predator_births = 0U;
  out_stats->prey_births = 0U;
  out_stats->predator_deaths = 0U;
  out_stats->prey_deaths = 0U;
  out_stats->kills = 0U;
  out_stats->food_eaten = 0U;
  out_stats->avg_predator_energy = predator_summary.avg_energy;
  out_stats->avg_prey_energy = prey_summary.avg_energy;
  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" std::int32_t moonai_gpu_evolution_innovation_state(const void *state_ptr,
                                                                DeviceInnovationState *out_innovation) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_innovation == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  return static_cast<std::int32_t>(
      moonai_gpu::copy_compact_device_readback(state->innovation, out_innovation, sizeof(*out_innovation)));
}

extern "C" std::int32_t moonai_gpu_last_cuda_error_code() { return moonai_gpu::g_last_cuda_error_code; }
