#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace moonai_gpu {

extern std::int32_t g_last_cuda_error_code;

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
constexpr std::uint8_t kHiddenNodeType = 1;
constexpr std::uint8_t kOutputNodeType = 2;
constexpr std::uint8_t kBiasNodeType = 3;
constexpr std::uint32_t kInnovationRecordAddConnection = 1U;
constexpr std::uint32_t kInnovationRecordAddNodeIncoming = 2U;
constexpr std::uint32_t kInnovationRecordAddNodeOutgoing = 3U;
constexpr std::uint32_t kCompileScratchNodeLimit = 128U;
constexpr std::uint32_t kCompileScratchConnectionLimit = 256U;
constexpr std::uint32_t kSpeciesBucketCount = 64U;
constexpr std::uint32_t kNearestTargetsPerType = 5U;
constexpr std::uint32_t kTargetCoordinateCount = 2U;
constexpr std::uint32_t kPerTypeSensorCount = kNearestTargetsPerType * kTargetCoordinateCount;
constexpr std::uint32_t kTargetTypeCount = 3U;
constexpr std::uint32_t kTargetSensorCount = kPerTypeSensorCount * kTargetTypeCount;
constexpr std::uint32_t kSelfStateSensorCount = 3U;
constexpr std::uint32_t kWallSensorCount = 2U;
constexpr std::uint32_t kSensorInputCount = kTargetSensorCount + kSelfStateSensorCount + kWallSensorCount;
constexpr std::uint32_t kSelfEnergyInputIndex = kTargetSensorCount;
constexpr std::uint32_t kVelocityXInputIndex = kSelfEnergyInputIndex + 1U;
constexpr std::uint32_t kVelocityYInputIndex = kSelfEnergyInputIndex + 2U;
constexpr std::uint32_t kWallXInputIndex = kSelfEnergyInputIndex + kSelfStateSensorCount;
constexpr std::uint32_t kWallYInputIndex = kWallXInputIndex + 1U;
constexpr std::uint32_t kUnclaimedMate = 0xFFFF'FFFFU;

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
  float *sensor_inputs;
  DeviceGenomeBuffers genome;
  DeviceCompiledNetworkBuffers compiled;
  std::uint32_t capacity;
};

struct FoodBuffer {
  float *pos_x;
  float *pos_y;
  std::uint8_t *active;
  std::uint32_t capacity;
};

struct DeviceInnovationState {
  std::uint32_t next_innovation;
  std::uint32_t next_node_id;
  std::uint32_t log_capacity;
  std::uint32_t log_len;
};

struct InnovationRecord {
  std::uint32_t from_node;
  std::uint32_t to_node;
  std::uint32_t innovation;
  std::uint32_t record_kind;
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

struct GpuSimulationConfig {
  std::uint32_t food_capacity;
  float world_size;
  float predator_speed;
  float prey_speed;
  float vision_range;
  float interaction_range;
  float mate_range;
  float energy_drain_per_tick;
  float energy_gain_from_kill;
  float energy_gain_from_food;
  float initial_energy;
  float max_energy;
  float reproduction_energy_threshold;
  float reproduction_energy_cost;
  float offspring_initial_energy;
  float mutation_rate;
  float weight_mutation_power;
  float add_node_rate;
  float add_connection_rate;
  float delete_connection_rate;
  std::uint32_t max_connection_attempts;
  std::uint32_t max_age;
  std::uint32_t report_interval_ticks;
  std::uint64_t seed;
};

struct GpuMutationConfig {
  float mutation_rate;
  float weight_mutation_power;
  float add_node_rate;
  float add_connection_rate;
  float delete_connection_rate;
  std::uint32_t max_connection_attempts;
};

struct MutationSummaryReadback {
  PopulationKind population_kind;
  std::uint32_t agents_mutated;
  std::uint32_t weight_perturbations;
  std::uint32_t added_connections;
  std::uint32_t added_nodes;
  std::uint32_t deleted_connections;
};

struct CrossoverSummaryReadback {
  PopulationKind population_kind;
  std::uint32_t parent_a_slot;
  std::uint32_t parent_b_slot;
  std::uint32_t offspring_slot;
  std::uint32_t offspring_entity_id;
  std::uint32_t offspring_generation;
  std::uint32_t inherited_connections;
  std::uint32_t matching_genes;
  std::uint32_t disjoint_genes;
  std::uint32_t excess_genes;
  std::uint64_t offspring_genome_hash;
};

struct SpeciesSummaryReadback {
  PopulationKind population_kind;
  std::uint32_t species_id;
  std::uint32_t size;
  std::uint32_t representative_slot;
  float avg_complexity;
};

struct RepresentativeGenomeHeader {
  PopulationKind population_kind;
  std::uint32_t slot;
  std::uint32_t entity_id;
  std::uint32_t generation;
  std::uint32_t species_id;
  std::uint16_t num_nodes;
  std::uint16_t num_connections;
};

struct GenomeNodeReadback {
  std::uint32_t id;
  std::uint8_t node_type;
  std::uint8_t reserved0;
  std::uint16_t reserved1;
};

struct GenomeConnectionReadback {
  std::int32_t from_node;
  std::int32_t to_node;
  float weight;
  std::uint32_t innovation;
  std::uint8_t enabled;
  std::uint8_t reserved0;
  std::uint16_t reserved1;
};

struct SpeciesBatchReadbackHeader {
  PopulationKind population_kind;
  std::uint32_t species_count;
  std::uint32_t returned_species_count;
};

struct CompiledNetworkReadbackHeader {
  PopulationKind population_kind;
  std::uint32_t slot;
  std::uint16_t node_count;
  std::uint16_t eval_node_count;
  std::uint16_t output_count;
  std::uint16_t connection_count;
};

struct SelectedAgentNetworkReadback {
  PopulationKind population_kind;
  std::uint32_t slot;
  std::uint16_t node_count;
  std::uint16_t output_count;
  std::uint16_t activation_count;
  std::uint16_t reserved;
  float output_0;
  float output_1;
};

struct SensorSnapshotReadback {
  PopulationKind population_kind;
  std::uint32_t slot;
  std::uint16_t input_count;
  std::uint16_t reserved;
  float inputs[kSensorInputCount];
};

struct RenderSnapshotHeader {
  std::uint32_t tick;
  std::uint32_t total_predators;
  std::uint32_t total_prey;
  std::uint32_t total_food;
  std::uint32_t returned_predators;
  std::uint32_t returned_prey;
  std::uint32_t returned_food;
};

struct RenderAgentReadback {
  PopulationKind population_kind;
  std::uint32_t slot;
  std::uint32_t entity_id;
  std::uint32_t species_id;
  std::uint32_t generation;
  float age;
  float pos_x;
  float pos_y;
  float dir_x;
  float dir_y;
  float energy;
};

struct RenderFoodReadback {
  std::uint32_t slot;
  std::uint8_t active;
  std::uint8_t reserved0;
  std::uint16_t reserved1;
  float pos_x;
  float pos_y;
};

struct ReproductionPair {
  std::uint32_t parent_a_slot;
  std::uint32_t parent_b_slot;
};

struct ReproductionSummaryReadback {
  PopulationKind population_kind;
  std::uint32_t eligible_parents;
  std::uint32_t candidate_pairs;
  std::uint32_t births;
  std::uint32_t failed_pairs;
};

struct FreeListStateReadback {
  std::uint32_t tick;
  std::uint32_t predator_free_slots;
  std::uint32_t prey_free_slots;
  std::uint32_t active_food_count;
  std::uint32_t food_capacity;
};

struct MetricsSummaryReadback {
  std::uint32_t tick;
  std::uint32_t predator_count;
  std::uint32_t prey_count;
  std::uint32_t predator_births;
  std::uint32_t prey_births;
  std::uint32_t predator_deaths;
  std::uint32_t prey_deaths;
  std::uint32_t predator_species;
  std::uint32_t prey_species;
  float avg_predator_complexity;
  float avg_prey_complexity;
  float avg_predator_energy;
  float avg_prey_energy;
  std::uint32_t max_predator_generation;
  float avg_predator_generation;
  std::uint32_t max_prey_generation;
  float avg_prey_generation;
};

struct SimulationCounters {
  std::uint32_t tick;
  std::uint32_t predator_births;
  std::uint32_t prey_births;
  std::uint32_t predator_deaths;
  std::uint32_t prey_deaths;
  std::uint32_t kills;
  std::uint32_t food_eaten;
};

struct PopulationGridEntry {
  std::uint32_t slot;
  float pos_x;
  float pos_y;
};

struct FoodGridEntry {
  std::uint32_t slot;
  float pos_x;
  float pos_y;
};

struct GpuEvolutionState {
  GpuEvolutionConfig config;
  GpuSimulationConfig simulation;
  DevicePopulationBuffers predator;
  DevicePopulationBuffers prey;
  FoodBuffer food;
  DeviceInnovationState *innovation;
  InnovationRecord *innovation_log;
  std::uint32_t *next_entity_id;
  SimulationCounters *counters;
  UiStatsReadback *mapped_ui_stats_host;
  UiStatsReadback *mapped_ui_stats_device;
  std::uint32_t *predator_free_list;
  std::uint32_t *prey_free_list;
  std::uint32_t *predator_free_len;
  std::uint32_t *prey_free_len;
  std::uint32_t *predator_mate_claims;
  std::uint32_t *prey_mate_claims;
  ReproductionPair *predator_reproduction_pairs;
  ReproductionPair *prey_reproduction_pairs;
  std::uint32_t *predator_pair_count;
  std::uint32_t *prey_pair_count;
  ReproductionSummaryReadback *predator_reproduction_summary;
  ReproductionSummaryReadback *prey_reproduction_summary;
  MetricsSummaryReadback *metrics_summary;
  std::uint32_t *predator_cell_counts;
  std::uint32_t *predator_cell_offsets;
  std::uint32_t *predator_cell_write_offsets;
  PopulationGridEntry *predator_grid_entries;
  std::uint32_t *prey_cell_counts;
  std::uint32_t *prey_cell_offsets;
  std::uint32_t *prey_cell_write_offsets;
  PopulationGridEntry *prey_grid_entries;
  std::uint32_t *food_cell_counts;
  std::uint32_t *food_cell_offsets;
  std::uint32_t *food_cell_write_offsets;
  FoodGridEntry *food_grid_entries;
  std::uint32_t grid_cols;
  std::uint32_t grid_rows;
  std::uint32_t grid_cell_capacity;
  float grid_cell_size;
};

inline bool is_runtime_unavailable_error(cudaError_t error) {
  return error == cudaErrorInsufficientDriver || error == cudaErrorInitializationError || error == cudaErrorNoDevice;
}

inline CudaStatus map_cuda_runtime_error(cudaError_t error, CudaStatus failure_status) {
  g_last_cuda_error_code = static_cast<std::int32_t>(error);
  if (is_runtime_unavailable_error(error)) {
    return CudaStatus::RuntimeUnavailable;
  }
  return error == cudaSuccess ? CudaStatus::Success : failure_status;
}

template <typename T> inline CudaStatus alloc_array(T **ptr, std::size_t count) {
  if (count == 0U) {
    *ptr = nullptr;
    return CudaStatus::Success;
  }

  void *raw = nullptr;
  const auto error = cudaMalloc(&raw, count * sizeof(T));
  const auto status = map_cuda_runtime_error(error, CudaStatus::AllocationFailed);
  if (status != CudaStatus::Success) {
    return status;
  }
  *ptr = static_cast<T *>(raw);
  return CudaStatus::Success;
}

template <typename T> inline void free_array(T *&ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

inline CudaStatus zero_device_memory(void *ptr, std::size_t size) {
  return map_cuda_runtime_error(cudaMemset(ptr, 0, size), CudaStatus::DeviceCopyFailed);
}

inline CudaStatus copy_compact_device_readback(const void *device_ptr, void *host_ptr, std::size_t size) {
  return map_cuda_runtime_error(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost), CudaStatus::DeviceCopyFailed);
}

inline CudaStatus copy_host_data_to_device(void *device_ptr, const void *host_ptr, std::size_t size) {
  return map_cuda_runtime_error(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice), CudaStatus::DeviceCopyFailed);
}

inline CudaStatus synchronize_kernels() {
  return map_cuda_runtime_error(cudaDeviceSynchronize(), CudaStatus::KernelLaunchFailed);
}

inline const DevicePopulationBuffers &population_for_kind(const GpuEvolutionState &state, PopulationKind population_kind) {
  return population_kind == PopulationKind::Predator ? state.predator : state.prey;
}

inline DevicePopulationBuffers &population_for_kind(GpuEvolutionState &state, PopulationKind population_kind) {
  return population_kind == PopulationKind::Predator ? state.predator : state.prey;
}

__device__ inline std::uint64_t splitmix64(std::uint64_t state) {
  state += 0x9e3779b97f4a7c15ULL;
  state = (state ^ (state >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  state = (state ^ (state >> 27U)) * 0x94d049bb133111ebULL;
  return state ^ (state >> 31U);
}

__device__ inline float next_unit_float(std::uint64_t &state) {
  state = splitmix64(state);
  const auto bits = static_cast<std::uint32_t>(state & 0x00FF'FFFFULL);
  return static_cast<float>(bits) / static_cast<float>(0x00FF'FFFFU);
}

__device__ inline float next_signed_float(std::uint64_t &state) { return (next_unit_float(state) * 2.0F) - 1.0F; }

__device__ inline std::uint64_t hash_mix(std::uint64_t hash, std::uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
  return hash;
}

__device__ inline void append_innovation_record(DeviceInnovationState *innovation, InnovationRecord *innovation_log,
                                                std::uint32_t from_node, std::uint32_t to_node,
                                                std::uint32_t innovation_id, std::uint32_t record_kind) {
  const auto log_index = atomicAdd(&innovation->log_len, 1U);
  if (log_index < innovation->log_capacity) {
    innovation_log[log_index] = InnovationRecord{from_node, to_node, innovation_id, record_kind};
  }
}

} // namespace moonai_gpu
