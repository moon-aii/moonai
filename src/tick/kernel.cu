#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>

namespace {

std::int32_t g_last_cuda_error_code = 0;

enum class CudaStatus : std::int32_t {
  Success = 0,
  InvalidArgument = 1,
  AllocationFailed = 2,
  KernelLaunchFailed = 3,
  DeviceCopyFailed = 4,
  RuntimeUnavailable = 5,
};

enum class PopulationKind : std::uint32_t {
  Predator = 0,
  Prey = 1,
};

constexpr std::uint8_t kInputNodeType = 0;
constexpr std::uint8_t kOutputNodeType = 2;
constexpr std::uint8_t kBiasNodeType = 3;

struct DeviceGenomeBuffers {
  std::int32_t *connection_from;
  std::int32_t *connection_to;
  float *connection_weight;
  std::uint32_t *connection_innovation;
  std::uint8_t *connection_enabled;
  std::uint8_t *node_types;
  std::uint16_t *num_connections;
  std::uint16_t *num_nodes;
  std::uint32_t connection_stride;
  std::uint32_t node_stride;
};

struct DeviceCompiledNetworkBuffers {
  std::uint16_t *eval_order;
  std::uint32_t *connection_offsets;
  std::uint16_t *output_indices;
  std::uint16_t *connection_sources;
  float *connection_weights;
  std::uint16_t *node_counts;
  std::uint16_t *eval_counts;
  std::uint16_t *connection_counts;
  std::uint32_t node_stride;
  std::uint32_t connection_stride;
  std::uint32_t output_stride;
};

struct DevicePopulationBuffers {
  float *pos_x;
  float *pos_y;
  float *vel_x;
  float *vel_y;
  float *energy;
  float *age;
  std::uint8_t *alive;
  std::uint32_t *species_id;
  std::uint32_t *entity_id;
  std::uint32_t *generation;
  std::uint64_t *rng_state;
  DeviceGenomeBuffers genome;
  DeviceCompiledNetworkBuffers compiled;
  std::uint32_t capacity;
};

struct DeviceInnovationState {
  std::uint32_t next_innovation;
  std::uint32_t next_node_id;
  std::uint32_t log_capacity;
  std::uint32_t log_len;
};

struct GpuEvolutionConfig {
  std::uint32_t predator_capacity;
  std::uint32_t prey_capacity;
  std::uint32_t initial_predator_count;
  std::uint32_t initial_prey_count;
  float world_size;
  float initial_energy;
  float max_energy;
  std::uint64_t seed;
  std::uint32_t num_inputs;
  std::uint32_t num_outputs;
  std::uint32_t node_stride;
  std::uint32_t connection_stride;
};

struct PopulationSummaryReadback {
  PopulationKind population_kind;
  std::uint32_t live_count;
  std::uint32_t capacity;
  std::uint32_t next_entity_id;
  std::uint32_t innovation_counter;
  std::uint32_t next_node_id;
  float avg_energy;
  float avg_connections;
};

struct SeededAgentSnapshot {
  PopulationKind population_kind;
  std::uint32_t slot;
  std::uint32_t entity_id;
  std::uint32_t generation;
  std::uint32_t species_id;
  std::uint8_t alive;
  std::uint8_t reserved0;
  std::uint16_t reserved1;
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
  float energy;
  float age;
  std::uint16_t num_nodes;
  std::uint16_t num_connections;
  std::uint64_t genome_hash;
};

struct UiStatsReadback {
  std::uint32_t tick;
  std::uint32_t predator_count;
  std::uint32_t prey_count;
  std::uint32_t predator_births;
  std::uint32_t prey_births;
  std::uint32_t predator_deaths;
  std::uint32_t prey_deaths;
  std::uint32_t kills;
  std::uint32_t food_eaten;
  float avg_predator_energy;
  float avg_prey_energy;
};

struct GpuEvolutionState {
  GpuEvolutionConfig config;
  DevicePopulationBuffers predator;
  DevicePopulationBuffers prey;
  DeviceInnovationState *innovation;
  std::uint32_t *next_entity_id;
};

template <typename T> CudaStatus alloc_array(T **ptr, std::size_t count) {
  if (count == 0) {
    *ptr = nullptr;
    return CudaStatus::Success;
  }

  void *raw = nullptr;
  const auto error = cudaMalloc(&raw, count * sizeof(T));
  g_last_cuda_error_code = static_cast<std::int32_t>(error);
  if (error == cudaErrorInsufficientDriver || error == cudaErrorInitializationError || error == cudaErrorNoDevice) {
    return CudaStatus::RuntimeUnavailable;
  }
  if (error != cudaSuccess) {
    return CudaStatus::AllocationFailed;
  }
  *ptr = static_cast<T *>(raw);
  return CudaStatus::Success;
}

template <typename T> void free_array(T *&ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

CudaStatus allocate_population_buffers(DevicePopulationBuffers &population, std::uint32_t capacity,
                                       std::uint32_t node_stride, std::uint32_t connection_stride,
                                       std::uint32_t output_stride) {
  population.capacity = capacity;
  population.genome.connection_stride = connection_stride;
  population.genome.node_stride = node_stride;
  population.compiled.node_stride = node_stride;
  population.compiled.connection_stride = connection_stride;
  population.compiled.output_stride = output_stride;

  if (alloc_array(&population.pos_x, capacity) != CudaStatus::Success ||
      alloc_array(&population.pos_y, capacity) != CudaStatus::Success ||
      alloc_array(&population.vel_x, capacity) != CudaStatus::Success ||
      alloc_array(&population.vel_y, capacity) != CudaStatus::Success ||
      alloc_array(&population.energy, capacity) != CudaStatus::Success ||
      alloc_array(&population.age, capacity) != CudaStatus::Success ||
      alloc_array(&population.alive, capacity) != CudaStatus::Success ||
      alloc_array(&population.species_id, capacity) != CudaStatus::Success ||
      alloc_array(&population.entity_id, capacity) != CudaStatus::Success ||
      alloc_array(&population.generation, capacity) != CudaStatus::Success ||
      alloc_array(&population.rng_state, capacity) != CudaStatus::Success ||
      alloc_array(&population.genome.connection_from, static_cast<std::size_t>(capacity) * connection_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.genome.connection_to, static_cast<std::size_t>(capacity) * connection_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.genome.connection_weight, static_cast<std::size_t>(capacity) * connection_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.genome.connection_innovation, static_cast<std::size_t>(capacity) * connection_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.genome.connection_enabled, static_cast<std::size_t>(capacity) * connection_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.genome.node_types, static_cast<std::size_t>(capacity) * node_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.genome.num_connections, capacity) != CudaStatus::Success ||
      alloc_array(&population.genome.num_nodes, capacity) != CudaStatus::Success ||
      alloc_array(&population.compiled.eval_order, static_cast<std::size_t>(capacity) * node_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.compiled.connection_offsets, static_cast<std::size_t>(capacity) * (node_stride + 1U)) !=
          CudaStatus::Success ||
      alloc_array(&population.compiled.output_indices, static_cast<std::size_t>(capacity) * output_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.compiled.connection_sources, static_cast<std::size_t>(capacity) * connection_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.compiled.connection_weights, static_cast<std::size_t>(capacity) * connection_stride) !=
          CudaStatus::Success ||
      alloc_array(&population.compiled.node_counts, capacity) != CudaStatus::Success ||
      alloc_array(&population.compiled.eval_counts, capacity) != CudaStatus::Success ||
      alloc_array(&population.compiled.connection_counts, capacity) != CudaStatus::Success) {
    return CudaStatus::AllocationFailed;
  }

  return CudaStatus::Success;
}

void free_population_buffers(DevicePopulationBuffers &population) {
  free_array(population.pos_x);
  free_array(population.pos_y);
  free_array(population.vel_x);
  free_array(population.vel_y);
  free_array(population.energy);
  free_array(population.age);
  free_array(population.alive);
  free_array(population.species_id);
  free_array(population.entity_id);
  free_array(population.generation);
  free_array(population.rng_state);
  free_array(population.genome.connection_from);
  free_array(population.genome.connection_to);
  free_array(population.genome.connection_weight);
  free_array(population.genome.connection_innovation);
  free_array(population.genome.connection_enabled);
  free_array(population.genome.node_types);
  free_array(population.genome.num_connections);
  free_array(population.genome.num_nodes);
  free_array(population.compiled.eval_order);
  free_array(population.compiled.connection_offsets);
  free_array(population.compiled.output_indices);
  free_array(population.compiled.connection_sources);
  free_array(population.compiled.connection_weights);
  free_array(population.compiled.node_counts);
  free_array(population.compiled.eval_counts);
  free_array(population.compiled.connection_counts);
  population.capacity = 0;
}

void destroy_state(GpuEvolutionState *state) {
  if (state == nullptr) {
    return;
  }
  free_population_buffers(state->predator);
  free_population_buffers(state->prey);
  free_array(state->innovation);
  free_array(state->next_entity_id);
  delete state;
}

__device__ std::uint64_t splitmix64(std::uint64_t state) {
  state += 0x9e3779b97f4a7c15ULL;
  state = (state ^ (state >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  state = (state ^ (state >> 27U)) * 0x94d049bb133111ebULL;
  return state ^ (state >> 31U);
}

__device__ float next_unit_float(std::uint64_t &state) {
  state = splitmix64(state);
  const auto bits = static_cast<std::uint32_t>(state & 0x00FF'FFFFULL);
  return static_cast<float>(bits) / static_cast<float>(0x00FF'FFFFU);
}

__device__ float next_signed_float(std::uint64_t &state) { return (next_unit_float(state) * 2.0F) - 1.0F; }

__device__ std::uint64_t hash_mix(std::uint64_t hash, std::uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
  return hash;
}

__global__ void seed_population_kernel(DevicePopulationBuffers population, std::uint32_t live_count,
                                       std::uint32_t base_entity_id, std::uint64_t base_seed,
                                       std::uint32_t num_inputs, std::uint32_t num_outputs, float world_size,
                                       float initial_energy, float max_energy, PopulationKind population_kind) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= live_count) {
    return;
  }

  std::uint64_t rng = base_seed ^ (static_cast<std::uint64_t>(population_kind == PopulationKind::Predator ? 0xA51U : 0xB29U)
                                   << 32U) ^ idx;
  rng = splitmix64(rng);

  population.pos_x[idx] = next_unit_float(rng) * world_size;
  population.pos_y[idx] = next_unit_float(rng) * world_size;
  population.vel_x[idx] = 0.0F;
  population.vel_y[idx] = 0.0F;
  population.energy[idx] = initial_energy < max_energy ? initial_energy : max_energy;
  population.age[idx] = 0.0F;
  population.alive[idx] = 1U;
  population.species_id[idx] = 0U;
  population.entity_id[idx] = base_entity_id + idx;
  population.generation[idx] = 0U;
  population.rng_state[idx] = rng;

  const auto node_count = static_cast<std::uint16_t>(num_inputs + num_outputs + 1U);
  population.genome.num_nodes[idx] = node_count;
  const auto seeded_connections = static_cast<std::uint16_t>((num_inputs + 1U) * num_outputs);
  population.genome.num_connections[idx] = seeded_connections;

  const auto node_base = static_cast<std::size_t>(idx) * population.genome.node_stride;
  for (std::uint32_t node = 0; node < population.genome.node_stride; ++node) {
    population.genome.node_types[node_base + node] = kInputNodeType;
  }
  for (std::uint32_t node = 0; node < num_inputs; ++node) {
    population.genome.node_types[node_base + node] = kInputNodeType;
  }
  population.genome.node_types[node_base + num_inputs] = kBiasNodeType;
  for (std::uint32_t node = 0; node < num_outputs; ++node) {
    population.genome.node_types[node_base + num_inputs + 1U + node] = kOutputNodeType;
  }

  const auto connection_base = static_cast<std::size_t>(idx) * population.genome.connection_stride;
  std::uint32_t connection_idx = 0;
  for (std::uint32_t in_node = 0; in_node < num_inputs + 1U; ++in_node) {
    for (std::uint32_t out_idx = 0; out_idx < num_outputs; ++out_idx) {
      const auto slot = connection_base + connection_idx;
      population.genome.connection_from[slot] = static_cast<std::int32_t>(in_node);
      population.genome.connection_to[slot] = static_cast<std::int32_t>(num_inputs + 1U + out_idx);
      population.genome.connection_weight[slot] = next_signed_float(rng);
      population.genome.connection_innovation[slot] = (in_node * num_outputs) + out_idx;
      population.genome.connection_enabled[slot] = 1U;
      ++connection_idx;
    }
  }

  const auto compiled_node_base = static_cast<std::size_t>(idx) * population.compiled.node_stride;
  const auto compiled_output_base = static_cast<std::size_t>(idx) * population.compiled.output_stride;
  const auto compiled_connection_base = static_cast<std::size_t>(idx) * population.compiled.connection_stride;
  const auto compiled_offset_base = static_cast<std::size_t>(idx) * (population.compiled.node_stride + 1U);
  population.compiled.node_counts[idx] = node_count;
  population.compiled.eval_counts[idx] = 0U;
  population.compiled.connection_counts[idx] = seeded_connections;
  for (std::uint32_t node = 0; node < population.compiled.node_stride; ++node) {
    population.compiled.eval_order[compiled_node_base + node] = static_cast<std::uint16_t>(node);
    population.compiled.connection_offsets[compiled_offset_base + node] = 0U;
  }
  population.compiled.connection_offsets[compiled_offset_base + population.compiled.node_stride] = seeded_connections;
  for (std::uint32_t output = 0; output < population.compiled.output_stride; ++output) {
    population.compiled.output_indices[compiled_output_base + output] =
        output < num_outputs ? static_cast<std::uint16_t>(num_inputs + 1U + output) : 0U;
  }
  for (std::uint32_t connection = 0; connection < population.compiled.connection_stride; ++connection) {
    const auto slot = compiled_connection_base + connection;
    if (connection < seeded_connections) {
      population.compiled.connection_sources[slot] =
          static_cast<std::uint16_t>(population.genome.connection_from[connection_base + connection]);
      population.compiled.connection_weights[slot] = population.genome.connection_weight[connection_base + connection];
    } else {
      population.compiled.connection_sources[slot] = 0U;
      population.compiled.connection_weights[slot] = 0.0F;
    }
  }
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
    genome_hash = hash_mix(genome_hash, population.genome.node_types[node_base + idx]);
  }

  const auto connection_base = static_cast<std::size_t>(slot) * population.genome.connection_stride;
  for (std::uint32_t idx = 0; idx < population.genome.num_connections[slot]; ++idx) {
    const auto entry = connection_base + idx;
    genome_hash = hash_mix(genome_hash, static_cast<std::uint32_t>(population.genome.connection_from[entry]));
    genome_hash = hash_mix(genome_hash, static_cast<std::uint32_t>(population.genome.connection_to[entry]));
    genome_hash = hash_mix(genome_hash, population.genome.connection_innovation[entry]);
    genome_hash = hash_mix(genome_hash, population.genome.connection_enabled[entry]);
    genome_hash = hash_mix(genome_hash, static_cast<std::uint64_t>(__float_as_uint(population.genome.connection_weight[entry])));
  }

  out_snapshot->genome_hash = genome_hash;
}

const DevicePopulationBuffers &population_for_kind(const GpuEvolutionState &state, PopulationKind population_kind) {
  return population_kind == PopulationKind::Predator ? state.predator : state.prey;
}

CudaStatus copy_compact_device_readback(const void *device_ptr, void *host_ptr, std::size_t size) {
  const auto error = cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);
  g_last_cuda_error_code = static_cast<std::int32_t>(error);
  if (error == cudaErrorInsufficientDriver || error == cudaErrorInitializationError || error == cudaErrorNoDevice) {
    return CudaStatus::RuntimeUnavailable;
  }
  return error == cudaSuccess ? CudaStatus::Success : CudaStatus::DeviceCopyFailed;
}

} // namespace

extern "C" std::int32_t moonai_gpu_runtime_available() {
  const auto error = cudaFree(nullptr);
  g_last_cuda_error_code = static_cast<std::int32_t>(error);
  if (error == cudaSuccess) {
    return static_cast<std::int32_t>(CudaStatus::Success);
  }
  if (error == cudaErrorInsufficientDriver || error == cudaErrorInitializationError || error == cudaErrorNoDevice) {
    return static_cast<std::int32_t>(CudaStatus::RuntimeUnavailable);
  }
  return static_cast<std::int32_t>(CudaStatus::KernelLaunchFailed);
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

  if (allocate_population_buffers(state->predator, config->predator_capacity, config->node_stride, config->connection_stride,
                                  config->num_outputs) != CudaStatus::Success ||
      allocate_population_buffers(state->prey, config->prey_capacity, config->node_stride, config->connection_stride,
                                  config->num_outputs) != CudaStatus::Success ||
      alloc_array(&state->innovation, 1U) != CudaStatus::Success ||
      alloc_array(&state->next_entity_id, 1U) != CudaStatus::Success) {
    destroy_state(state);
    return static_cast<std::int32_t>(CudaStatus::AllocationFailed);
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

  const auto predator_blocks = (state->config.initial_predator_count + 255U) / 256U;
  const auto prey_blocks = (state->config.initial_prey_count + 255U) / 256U;
  seed_population_kernel<<<predator_blocks, 256U>>>(
      state->predator, state->config.initial_predator_count, 1U, state->config.seed, state->config.num_inputs,
      state->config.num_outputs, state->config.world_size, state->config.initial_energy, state->config.max_energy,
      PopulationKind::Predator);
  seed_population_kernel<<<prey_blocks, 256U>>>(
      state->prey, state->config.initial_prey_count, state->config.initial_predator_count + 1U,
      state->config.seed ^ 0x9e3779b97f4a7c15ULL, state->config.num_inputs, state->config.num_outputs,
      state->config.world_size, state->config.initial_energy, state->config.max_energy, PopulationKind::Prey);
  const auto sync_error = cudaDeviceSynchronize();
  g_last_cuda_error_code = static_cast<std::int32_t>(sync_error);
  if (sync_error != cudaSuccess) {
    return static_cast<std::int32_t>(CudaStatus::KernelLaunchFailed);
  }

  DeviceInnovationState innovation_state{};
  innovation_state.next_innovation = (state->config.num_inputs + 1U) * state->config.num_outputs;
  innovation_state.next_node_id = state->config.num_inputs + state->config.num_outputs + 1U;
  innovation_state.log_capacity = 0U;
  innovation_state.log_len = 0U;
  const auto innovation_copy = cudaMemcpy(state->innovation, &innovation_state, sizeof(innovation_state), cudaMemcpyHostToDevice);
  g_last_cuda_error_code = static_cast<std::int32_t>(innovation_copy);
  if (innovation_copy != cudaSuccess) {
    return static_cast<std::int32_t>(CudaStatus::DeviceCopyFailed);
  }

  const std::uint32_t next_entity_id = state->config.initial_predator_count + state->config.initial_prey_count + 1U;
  const auto entity_copy = cudaMemcpy(state->next_entity_id, &next_entity_id, sizeof(next_entity_id), cudaMemcpyHostToDevice);
  g_last_cuda_error_code = static_cast<std::int32_t>(entity_copy);
  if (entity_copy != cudaSuccess) {
    return static_cast<std::int32_t>(CudaStatus::DeviceCopyFailed);
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
  if (alloc_array(&device_summary, 1U) != CudaStatus::Success) {
    return static_cast<std::int32_t>(CudaStatus::AllocationFailed);
  }

  summarize_population_kernel<<<1U, 1U>>>(population_for_kind(*state, population_kind), state->innovation,
                                          state->next_entity_id, population_kind, device_summary);
  const auto sync_error = cudaDeviceSynchronize();
  g_last_cuda_error_code = static_cast<std::int32_t>(sync_error);
  if (sync_error != cudaSuccess) {
    free_array(device_summary);
    return static_cast<std::int32_t>(CudaStatus::KernelLaunchFailed);
  }

  const auto status = copy_compact_device_readback(device_summary, out_summary, sizeof(*out_summary));
  free_array(device_summary);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_seeded_agent_snapshot(const void *state_ptr, PopulationKind population_kind,
                                                                     std::uint32_t slot,
                                                                     SeededAgentSnapshot *out_snapshot) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_snapshot == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto &population = population_for_kind(*state, population_kind);
  if (slot >= population.capacity) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  SeededAgentSnapshot *device_snapshot = nullptr;
  if (alloc_array(&device_snapshot, 1U) != CudaStatus::Success) {
    return static_cast<std::int32_t>(CudaStatus::AllocationFailed);
  }

  seeded_agent_snapshot_kernel<<<1U, 1U>>>(population, population_kind, slot, device_snapshot);
  const auto sync_error = cudaDeviceSynchronize();
  g_last_cuda_error_code = static_cast<std::int32_t>(sync_error);
  if (sync_error != cudaSuccess) {
    free_array(device_snapshot);
    return static_cast<std::int32_t>(CudaStatus::KernelLaunchFailed);
  }

  const auto status = copy_compact_device_readback(device_snapshot, out_snapshot, sizeof(*out_snapshot));
  free_array(device_snapshot);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_evolution_ui_stats(const void *state_ptr, UiStatsReadback *out_stats) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_stats == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  PopulationSummaryReadback predator_summary{};
  PopulationSummaryReadback prey_summary{};
  const auto predator_status = static_cast<CudaStatus>(moonai_gpu_evolution_population_summary(
      state_ptr, PopulationKind::Predator, &predator_summary));
  if (predator_status != CudaStatus::Success) {
    return static_cast<std::int32_t>(predator_status);
  }

  const auto prey_status = static_cast<CudaStatus>(moonai_gpu_evolution_population_summary(
      state_ptr, PopulationKind::Prey, &prey_summary));
  if (prey_status != CudaStatus::Success) {
    return static_cast<std::int32_t>(prey_status);
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

  return static_cast<std::int32_t>(copy_compact_device_readback(state->innovation, out_innovation, sizeof(*out_innovation)));
}

extern "C" std::int32_t moonai_gpu_last_cuda_error_code() { return g_last_cuda_error_code; }
