#include "evolution_cuda.cuh"

#include <new>
#include <tuple>
#include <vector>

using moonai_gpu::CudaStatus;
using moonai_gpu::DeviceInnovationState;
using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::FoodBuffer;
using moonai_gpu::GpuEvolutionConfig;
using moonai_gpu::GpuEvolutionState;
using moonai_gpu::GpuMutationConfig;
using moonai_gpu::GpuSimulationConfig;
using moonai_gpu::InnovationRecord;
using moonai_gpu::PopulationKind;
using moonai_gpu::PopulationSummaryReadback;
using moonai_gpu::RenderAgentReadback;
using moonai_gpu::RenderFoodReadback;
using moonai_gpu::RenderSnapshotHeader;
using moonai_gpu::ReproductionPair;
using moonai_gpu::ReproductionSummaryReadback;
using moonai_gpu::SensorSnapshotReadback;
using moonai_gpu::SimulationCounters;
using moonai_gpu::UiStatsReadback;
using moonai_gpu::PopulationGridEntry;
using moonai_gpu::FoodGridEntry;
using moonai_gpu::MetricsSummaryReadback;

namespace moonai_gpu {

std::int32_t g_last_cuda_error_code = 0;

} // namespace moonai_gpu

extern "C" std::int32_t moonai_gpu_evolution_crossover(void *state_ptr, PopulationKind population_kind,
                                                          std::uint32_t parent_a_slot, std::uint32_t parent_b_slot,
                                                          std::uint32_t offspring_slot,
                                                          moonai_gpu::CrossoverSummaryReadback *out_summary);
extern "C" std::int32_t moonai_gpu_evolution_population_summary(const void *state_ptr, PopulationKind population_kind,
                                                                  moonai_gpu::PopulationSummaryReadback *out_summary);
extern "C" std::int32_t moonai_gpu_evolution_mutate_slot(void *state_ptr, PopulationKind population_kind,
                                                             std::uint32_t slot,
                                                             const moonai_gpu::GpuMutationConfig *config,
                                                             moonai_gpu::MutationSummaryReadback *out_summary);
extern "C" std::int32_t moonai_gpu_evolution_compile_slot(void *state_ptr, PopulationKind population_kind,
                                                              std::uint32_t slot,
                                                              moonai_gpu::CompiledNetworkReadbackHeader *out_header);
extern "C" std::int32_t moonai_gpu_evolution_species_summaries(void *state_ptr, PopulationKind population_kind,
                                                                   std::uint32_t max_species,
                                                                   moonai_gpu::SpeciesBatchReadbackHeader *out_header,
                                                                   moonai_gpu::SpeciesSummaryReadback *out_summaries,
                                                                   moonai_gpu::RepresentativeGenomeHeader *out_representatives);

namespace {

__global__ void initialize_free_list_kernel(DevicePopulationBuffers population, std::uint32_t *free_list,
                                            std::uint32_t *free_len);

CudaStatus allocate_population_buffers(DevicePopulationBuffers &population, std::uint32_t capacity,
                                       std::uint32_t num_inputs, std::uint32_t node_stride,
                                       std::uint32_t connection_stride, std::uint32_t output_stride) {
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
            moonai_gpu::alloc_array(&population.sensor_inputs, static_cast<std::size_t>(capacity) * num_inputs),
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
  moonai_gpu::free_array(population.sensor_inputs);
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

CudaStatus allocate_food_buffers(FoodBuffer &food, std::uint32_t capacity) {
  food.capacity = capacity;
  for (auto status : {moonai_gpu::alloc_array(&food.pos_x, capacity), moonai_gpu::alloc_array(&food.pos_y, capacity),
                      moonai_gpu::alloc_array(&food.active, capacity)}) {
    if (status != CudaStatus::Success) {
      return status;
    }
  }
  return CudaStatus::Success;
}

void free_food_buffers(FoodBuffer &food) {
  moonai_gpu::free_array(food.pos_x);
  moonai_gpu::free_array(food.pos_y);
  moonai_gpu::free_array(food.active);
  food.capacity = 0U;
}

CudaStatus allocate_mapped_ui_stats(UiStatsReadback **host_ptr, UiStatsReadback **device_ptr) {
  void *mapped_host = nullptr;
  const auto alloc_error = cudaHostAlloc(&mapped_host, sizeof(UiStatsReadback), cudaHostAllocMapped);
  auto status = moonai_gpu::map_cuda_runtime_error(alloc_error, CudaStatus::AllocationFailed);
  if (status != CudaStatus::Success) {
    return status;
  }

  void *mapped_device = nullptr;
  const auto device_error = cudaHostGetDevicePointer(&mapped_device, mapped_host, 0U);
  status = moonai_gpu::map_cuda_runtime_error(device_error, CudaStatus::AllocationFailed);
  if (status != CudaStatus::Success) {
    cudaFreeHost(mapped_host);
    return status;
  }

  *host_ptr = static_cast<UiStatsReadback *>(mapped_host);
  *device_ptr = static_cast<UiStatsReadback *>(mapped_device);
  **host_ptr = UiStatsReadback{};
  return CudaStatus::Success;
}

void free_mapped_ui_stats(UiStatsReadback *&host_ptr, UiStatsReadback *&device_ptr) {
  if (host_ptr != nullptr) {
    cudaFreeHost(host_ptr);
  }
  host_ptr = nullptr;
  device_ptr = nullptr;
}

void free_spatial_grid_buffers(GpuEvolutionState &state) {
  moonai_gpu::free_array(state.predator_cell_counts);
  moonai_gpu::free_array(state.predator_cell_offsets);
  moonai_gpu::free_array(state.predator_cell_write_offsets);
  moonai_gpu::free_array(state.predator_grid_entries);
  moonai_gpu::free_array(state.prey_cell_counts);
  moonai_gpu::free_array(state.prey_cell_offsets);
  moonai_gpu::free_array(state.prey_cell_write_offsets);
  moonai_gpu::free_array(state.prey_grid_entries);
  moonai_gpu::free_array(state.food_cell_counts);
  moonai_gpu::free_array(state.food_cell_offsets);
  moonai_gpu::free_array(state.food_cell_write_offsets);
  moonai_gpu::free_array(state.food_grid_entries);
  state.grid_cols = 0U;
  state.grid_rows = 0U;
  state.grid_cell_capacity = 0U;
  state.grid_cell_size = 0.0F;
}

void free_reproduction_buffers(GpuEvolutionState &state) {
  moonai_gpu::free_array(state.predator_mate_claims);
  moonai_gpu::free_array(state.prey_mate_claims);
  moonai_gpu::free_array(state.predator_reproduction_pairs);
  moonai_gpu::free_array(state.prey_reproduction_pairs);
  moonai_gpu::free_array(state.predator_pair_count);
  moonai_gpu::free_array(state.prey_pair_count);
  moonai_gpu::free_array(state.predator_reproduction_summary);
  moonai_gpu::free_array(state.prey_reproduction_summary);
}

CudaStatus ensure_reproduction_buffers(GpuEvolutionState &state) {
  if (state.predator_mate_claims != nullptr && state.prey_mate_claims != nullptr && state.predator_pair_count != nullptr &&
      state.prey_pair_count != nullptr && state.predator_reproduction_summary != nullptr &&
      state.prey_reproduction_summary != nullptr && (state.predator.capacity == 0U || state.predator_reproduction_pairs != nullptr) &&
      (state.prey.capacity == 0U || state.prey_reproduction_pairs != nullptr)) {
    return CudaStatus::Success;
  }

  free_reproduction_buffers(state);
  for (auto status : {
           moonai_gpu::alloc_array(&state.predator_mate_claims, state.predator.capacity),
           moonai_gpu::alloc_array(&state.prey_mate_claims, state.prey.capacity),
           moonai_gpu::alloc_array(&state.predator_reproduction_pairs, state.predator.capacity),
           moonai_gpu::alloc_array(&state.prey_reproduction_pairs, state.prey.capacity),
           moonai_gpu::alloc_array(&state.predator_pair_count, 1U),
           moonai_gpu::alloc_array(&state.prey_pair_count, 1U),
           moonai_gpu::alloc_array(&state.predator_reproduction_summary, 1U),
           moonai_gpu::alloc_array(&state.prey_reproduction_summary, 1U),
       }) {
    if (status != CudaStatus::Success) {
      free_reproduction_buffers(state);
      return status;
    }
  }
  return CudaStatus::Success;
}

CudaStatus ensure_metrics_buffer(GpuEvolutionState &state) {
  if (state.metrics_summary != nullptr) {
    return CudaStatus::Success;
  }
  return moonai_gpu::alloc_array(&state.metrics_summary, 1U);
}

CudaStatus ensure_spatial_grid_buffers(GpuEvolutionState &state, std::uint32_t cell_count) {
  if (cell_count == 0U) {
    free_spatial_grid_buffers(state);
    return CudaStatus::Success;
  }
  if (state.grid_cell_capacity == cell_count && state.predator_cell_counts != nullptr &&
      state.predator_cell_offsets != nullptr && state.predator_cell_write_offsets != nullptr &&
      (state.predator.capacity == 0U || state.predator_grid_entries != nullptr) &&
      state.prey_cell_counts != nullptr && state.prey_cell_offsets != nullptr &&
      state.prey_cell_write_offsets != nullptr && (state.prey.capacity == 0U || state.prey_grid_entries != nullptr) &&
      state.food_cell_counts != nullptr && state.food_cell_offsets != nullptr &&
      state.food_cell_write_offsets != nullptr && (state.food.capacity == 0U || state.food_grid_entries != nullptr)) {
    return CudaStatus::Success;
  }

  const auto grid_cols = state.grid_cols;
  const auto grid_rows = state.grid_rows;
  const auto grid_cell_size = state.grid_cell_size;
  free_spatial_grid_buffers(state);
  state.grid_cols = grid_cols;
  state.grid_rows = grid_rows;
  state.grid_cell_size = grid_cell_size;
  for (auto status : {
           moonai_gpu::alloc_array(&state.predator_cell_counts, cell_count),
           moonai_gpu::alloc_array(&state.predator_cell_offsets, cell_count + 1U),
           moonai_gpu::alloc_array(&state.predator_cell_write_offsets, cell_count),
           moonai_gpu::alloc_array(&state.predator_grid_entries, state.predator.capacity),
           moonai_gpu::alloc_array(&state.prey_cell_counts, cell_count),
           moonai_gpu::alloc_array(&state.prey_cell_offsets, cell_count + 1U),
           moonai_gpu::alloc_array(&state.prey_cell_write_offsets, cell_count),
           moonai_gpu::alloc_array(&state.prey_grid_entries, state.prey.capacity),
           moonai_gpu::alloc_array(&state.food_cell_counts, cell_count),
           moonai_gpu::alloc_array(&state.food_cell_offsets, cell_count + 1U),
           moonai_gpu::alloc_array(&state.food_cell_write_offsets, cell_count),
           moonai_gpu::alloc_array(&state.food_grid_entries, state.food.capacity),
       }) {
    if (status != CudaStatus::Success) {
      free_spatial_grid_buffers(state);
      return status;
    }
  }

  state.grid_cell_capacity = cell_count;
  return CudaStatus::Success;
}

void destroy_state(GpuEvolutionState *state) {
  if (state == nullptr) {
    return;
  }
  free_food_buffers(state->food);
  free_population_buffers(state->predator);
  free_population_buffers(state->prey);
  moonai_gpu::free_array(state->innovation);
  moonai_gpu::free_array(state->innovation_log);
  moonai_gpu::free_array(state->next_entity_id);
  moonai_gpu::free_array(state->counters);
  moonai_gpu::free_array(state->predator_free_list);
  moonai_gpu::free_array(state->prey_free_list);
  moonai_gpu::free_array(state->predator_free_len);
  moonai_gpu::free_array(state->prey_free_len);
  free_reproduction_buffers(*state);
  moonai_gpu::free_array(state->metrics_summary);
  free_spatial_grid_buffers(*state);
  free_mapped_ui_stats(state->mapped_ui_stats_host, state->mapped_ui_stats_device);
  delete state;
}

CudaStatus device_copy_bytes(void *dst, const void *src, std::size_t size) {
  return moonai_gpu::map_cuda_runtime_error(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice),
                                            CudaStatus::DeviceCopyFailed);
}

CudaStatus copy_population_buffers(const DevicePopulationBuffers &src, DevicePopulationBuffers &dst,
                                   std::uint32_t num_inputs) {
  const auto copy_cap = src.capacity;
  const auto connection_bytes = static_cast<std::size_t>(copy_cap) * src.genome.connection_stride;
  const auto node_bytes = static_cast<std::size_t>(copy_cap) * src.genome.node_stride;
  const auto offset_bytes = static_cast<std::size_t>(copy_cap) * (src.compiled.node_stride + 1U);
  const auto output_bytes = static_cast<std::size_t>(copy_cap) * src.compiled.output_stride;

  for (const auto [dst_ptr, src_ptr, size] : {
           std::tuple<void *, const void *, std::size_t>{dst.pos_x, src.pos_x, sizeof(float) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.pos_y, src.pos_y, sizeof(float) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.vel_x, src.vel_x, sizeof(float) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.vel_y, src.vel_y, sizeof(float) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.energy, src.energy, sizeof(float) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.age, src.age, sizeof(float) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.alive, src.alive, sizeof(std::uint8_t) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.species_id, src.species_id, sizeof(std::uint32_t) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.entity_id, src.entity_id, sizeof(std::uint32_t) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.generation, src.generation, sizeof(std::uint32_t) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.rng_state, src.rng_state, sizeof(std::uint64_t) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.sensor_inputs, src.sensor_inputs,
                                                         sizeof(float) * static_cast<std::size_t>(copy_cap) * num_inputs},
           std::tuple<void *, const void *, std::size_t>{dst.genome.connection_from, src.genome.connection_from,
                                                         sizeof(std::int32_t) * connection_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.genome.connection_to, src.genome.connection_to,
                                                         sizeof(std::int32_t) * connection_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.genome.connection_weight, src.genome.connection_weight,
                                                         sizeof(float) * connection_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.genome.connection_innovation,
                                                         src.genome.connection_innovation,
                                                         sizeof(std::uint32_t) * connection_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.genome.connection_enabled, src.genome.connection_enabled,
                                                         sizeof(std::uint8_t) * connection_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.genome.node_types, src.genome.node_types,
                                                         sizeof(std::uint8_t) * node_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.genome.num_connections, src.genome.num_connections,
                                                         sizeof(std::uint16_t) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.genome.num_nodes, src.genome.num_nodes,
                                                         sizeof(std::uint16_t) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.compiled.eval_order, src.compiled.eval_order,
                                                         sizeof(std::uint16_t) * node_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.compiled.connection_offsets, src.compiled.connection_offsets,
                                                         sizeof(std::uint32_t) * offset_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.compiled.output_indices, src.compiled.output_indices,
                                                         sizeof(std::uint16_t) * output_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.compiled.connection_sources,
                                                         src.compiled.connection_sources,
                                                         sizeof(std::uint16_t) * connection_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.compiled.connection_weights,
                                                         src.compiled.connection_weights,
                                                         sizeof(float) * connection_bytes},
           std::tuple<void *, const void *, std::size_t>{dst.compiled.node_counts, src.compiled.node_counts,
                                                         sizeof(std::uint16_t) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.compiled.eval_counts, src.compiled.eval_counts,
                                                         sizeof(std::uint16_t) * copy_cap},
           std::tuple<void *, const void *, std::size_t>{dst.compiled.connection_counts,
                                                         src.compiled.connection_counts,
                                                         sizeof(std::uint16_t) * copy_cap},
       }) {
    const auto status = device_copy_bytes(dst_ptr, src_ptr, size);
    if (status != CudaStatus::Success) {
      return status;
    }
  }

  return CudaStatus::Success;
}

CudaStatus zero_population_tail(const DevicePopulationBuffers &population, std::uint32_t from_capacity,
                                std::uint32_t num_inputs) {
  if (from_capacity >= population.capacity) {
    return CudaStatus::Success;
  }

  const auto tail = population.capacity - from_capacity;
  const auto tail_connection_entries = static_cast<std::size_t>(tail) * population.genome.connection_stride;
  const auto tail_node_entries = static_cast<std::size_t>(tail) * population.genome.node_stride;
  const auto tail_offset_entries = static_cast<std::size_t>(tail) * (population.compiled.node_stride + 1U);
  const auto tail_output_entries = static_cast<std::size_t>(tail) * population.compiled.output_stride;

  for (const auto [ptr, size] : {
           std::pair<void *, std::size_t>{population.pos_x + from_capacity, sizeof(float) * tail},
           std::pair<void *, std::size_t>{population.pos_y + from_capacity, sizeof(float) * tail},
           std::pair<void *, std::size_t>{population.vel_x + from_capacity, sizeof(float) * tail},
           std::pair<void *, std::size_t>{population.vel_y + from_capacity, sizeof(float) * tail},
           std::pair<void *, std::size_t>{population.energy + from_capacity, sizeof(float) * tail},
           std::pair<void *, std::size_t>{population.age + from_capacity, sizeof(float) * tail},
           std::pair<void *, std::size_t>{population.alive + from_capacity, sizeof(std::uint8_t) * tail},
           std::pair<void *, std::size_t>{population.species_id + from_capacity, sizeof(std::uint32_t) * tail},
           std::pair<void *, std::size_t>{population.entity_id + from_capacity, sizeof(std::uint32_t) * tail},
           std::pair<void *, std::size_t>{population.generation + from_capacity, sizeof(std::uint32_t) * tail},
           std::pair<void *, std::size_t>{population.rng_state + from_capacity, sizeof(std::uint64_t) * tail},
           std::pair<void *, std::size_t>{population.sensor_inputs + (static_cast<std::size_t>(from_capacity) * num_inputs),
                                          sizeof(float) * static_cast<std::size_t>(tail) * num_inputs},
           std::pair<void *, std::size_t>{population.genome.connection_from +
                                              (static_cast<std::size_t>(from_capacity) * population.genome.connection_stride),
                                          sizeof(std::int32_t) * tail_connection_entries},
           std::pair<void *, std::size_t>{population.genome.connection_to +
                                              (static_cast<std::size_t>(from_capacity) * population.genome.connection_stride),
                                          sizeof(std::int32_t) * tail_connection_entries},
           std::pair<void *, std::size_t>{population.genome.connection_weight +
                                              (static_cast<std::size_t>(from_capacity) * population.genome.connection_stride),
                                          sizeof(float) * tail_connection_entries},
           std::pair<void *, std::size_t>{population.genome.connection_innovation +
                                              (static_cast<std::size_t>(from_capacity) * population.genome.connection_stride),
                                          sizeof(std::uint32_t) * tail_connection_entries},
           std::pair<void *, std::size_t>{population.genome.connection_enabled +
                                              (static_cast<std::size_t>(from_capacity) * population.genome.connection_stride),
                                          sizeof(std::uint8_t) * tail_connection_entries},
           std::pair<void *, std::size_t>{population.genome.node_types +
                                              (static_cast<std::size_t>(from_capacity) * population.genome.node_stride),
                                          sizeof(std::uint8_t) * tail_node_entries},
           std::pair<void *, std::size_t>{population.genome.num_connections + from_capacity,
                                          sizeof(std::uint16_t) * tail},
           std::pair<void *, std::size_t>{population.genome.num_nodes + from_capacity, sizeof(std::uint16_t) * tail},
           std::pair<void *, std::size_t>{population.compiled.eval_order +
                                              (static_cast<std::size_t>(from_capacity) * population.compiled.node_stride),
                                          sizeof(std::uint16_t) * tail_node_entries},
           std::pair<void *, std::size_t>{population.compiled.connection_offsets +
                                              (static_cast<std::size_t>(from_capacity) * (population.compiled.node_stride + 1U)),
                                          sizeof(std::uint32_t) * tail_offset_entries},
           std::pair<void *, std::size_t>{population.compiled.output_indices +
                                              (static_cast<std::size_t>(from_capacity) * population.compiled.output_stride),
                                          sizeof(std::uint16_t) * tail_output_entries},
           std::pair<void *, std::size_t>{population.compiled.connection_sources +
                                              (static_cast<std::size_t>(from_capacity) * population.compiled.connection_stride),
                                          sizeof(std::uint16_t) * tail_connection_entries},
           std::pair<void *, std::size_t>{population.compiled.connection_weights +
                                              (static_cast<std::size_t>(from_capacity) * population.compiled.connection_stride),
                                          sizeof(float) * tail_connection_entries},
           std::pair<void *, std::size_t>{population.compiled.node_counts + from_capacity,
                                          sizeof(std::uint16_t) * tail},
           std::pair<void *, std::size_t>{population.compiled.eval_counts + from_capacity,
                                          sizeof(std::uint16_t) * tail},
           std::pair<void *, std::size_t>{population.compiled.connection_counts + from_capacity,
                                          sizeof(std::uint16_t) * tail},
       }) {
    const auto status = moonai_gpu::zero_device_memory(ptr, size);
    if (status != CudaStatus::Success) {
      return status;
    }
  }

  return CudaStatus::Success;
}

CudaStatus replace_population_buffers(DevicePopulationBuffers &population, DevicePopulationBuffers replacement,
                                      std::uint32_t num_inputs) {
  const auto status = zero_population_tail(replacement, population.capacity, num_inputs);
  if (status != CudaStatus::Success) {
    free_population_buffers(replacement);
    return status;
  }
  free_population_buffers(population);
  population = replacement;
  return CudaStatus::Success;
}

CudaStatus expand_population_capacity(GpuEvolutionState &state, PopulationKind population_kind, std::uint32_t new_capacity) {
  auto &population = moonai_gpu::population_for_kind(state, population_kind);
  if (new_capacity <= population.capacity) {
    return CudaStatus::Success;
  }

  DevicePopulationBuffers replacement{};
  auto status = allocate_population_buffers(replacement, new_capacity, state.config.num_inputs, population.genome.node_stride,
                                            population.genome.connection_stride, population.compiled.output_stride);
  if (status != CudaStatus::Success) {
    return status;
  }

  status = copy_population_buffers(population, replacement, state.config.num_inputs);
  if (status != CudaStatus::Success) {
    free_population_buffers(replacement);
    return status;
  }
  status = replace_population_buffers(population, replacement, state.config.num_inputs);
  if (status != CudaStatus::Success) {
    return status;
  }

  const auto grid_cols = state.grid_cols;
  const auto grid_rows = state.grid_rows;
  const auto grid_cell_size = state.grid_cell_size;

  if (population_kind == PopulationKind::Predator) {
    moonai_gpu::free_array(state.predator_free_list);
  } else {
    moonai_gpu::free_array(state.prey_free_list);
  }
  free_reproduction_buffers(state);
  free_spatial_grid_buffers(state);
  state.grid_cols = grid_cols;
  state.grid_rows = grid_rows;
  state.grid_cell_size = grid_cell_size;

  if (population_kind == PopulationKind::Predator) {
    status = moonai_gpu::alloc_array(&state.predator_free_list, population.capacity);
  } else {
    status = moonai_gpu::alloc_array(&state.prey_free_list, population.capacity);
  }
  if (status != CudaStatus::Success) {
    return status;
  }

  auto *free_list = population_kind == PopulationKind::Predator ? state.predator_free_list : state.prey_free_list;
  auto *free_len = population_kind == PopulationKind::Predator ? state.predator_free_len : state.prey_free_len;
  initialize_free_list_kernel<<<1U, 1U>>>(population, free_list, free_len);
  status = moonai_gpu::synchronize_kernels();
  if (status != CudaStatus::Success) {
    return status;
  }

  status = ensure_reproduction_buffers(state);
  if (status != CudaStatus::Success) {
    return status;
  }
  return ensure_spatial_grid_buffers(state, state.grid_cols * state.grid_rows);
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
  const auto sensor_base = static_cast<std::size_t>(idx) * num_inputs;
  for (std::uint32_t sensor = 0; sensor < num_inputs; ++sensor) {
    population.sensor_inputs[sensor_base + sensor] = 0.0F;
  }

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

__device__ float clamp_world(float value, float world_size) {
  if (value < 0.0F) {
    return 0.0F;
  }
  if (value > world_size) {
    return world_size;
  }
  return value;
}

__device__ float clamp_range(float value, float min_value, float max_value) {
  return fminf(fmaxf(value, min_value), max_value);
}

template <std::uint32_t N>
__device__ void insert_nearest_candidate(float dx, float dy, float dist_sq, float (&best_dx)[N], float (&best_dy)[N],
                                         float (&best_dist_sq)[N]) {
  if (dist_sq >= best_dist_sq[N - 1U]) {
    return;
  }

  std::uint32_t insert_at = N - 1U;
  while (insert_at > 0U && dist_sq < best_dist_sq[insert_at - 1U]) {
    best_dist_sq[insert_at] = best_dist_sq[insert_at - 1U];
    best_dx[insert_at] = best_dx[insert_at - 1U];
    best_dy[insert_at] = best_dy[insert_at - 1U];
    --insert_at;
  }

  best_dist_sq[insert_at] = dist_sq;
  best_dx[insert_at] = dx;
  best_dy[insert_at] = dy;
}

template <std::uint32_t N>
__device__ void encode_nearest_targets(const float (&best_dx)[N], const float (&best_dy)[N],
                                       const float (&best_dist_sq)[N], float vision_range, float *out,
                                       std::uint32_t output_capacity) {
  for (std::uint32_t idx = 0; idx < N; ++idx) {
    const auto offset = idx * 2U;
    if (offset + 1U >= output_capacity) {
      return;
    }
    if (best_dist_sq[idx] == INFINITY) {
      out[offset] = 0.0F;
      out[offset + 1U] = 0.0F;
      continue;
    }

    const auto dist = sqrtf(best_dist_sq[idx]);
    if (dist <= 1e-6F) {
      out[offset] = 0.0F;
      out[offset + 1U] = 0.0F;
      continue;
    }

    const auto proximity = clamp_range(1.0F - (dist / vision_range), 0.0F, 1.0F);
    const auto inv_dist = 1.0F / dist;
    out[offset] = clamp_range(best_dx[idx] * inv_dist * proximity, -1.0F, 1.0F);
    out[offset + 1U] = clamp_range(best_dy[idx] * inv_dist * proximity, -1.0F, 1.0F);
  }
}

__device__ float encode_axis_wall_sensor(float negative_side_dist, float positive_side_dist, float vision_range) {
  const auto negative_in_range = negative_side_dist < vision_range;
  const auto positive_in_range = positive_side_dist < vision_range;
  if (!negative_in_range && !positive_in_range) {
    return 0.0F;
  }
  if (negative_in_range && (!positive_in_range || negative_side_dist <= positive_side_dist)) {
    return -(1.0F - (negative_side_dist / vision_range));
  }
  return 1.0F - (positive_side_dist / vision_range);
}

__device__ std::uint32_t cell_coord(float pos, float cell_size, std::uint32_t limit) {
  const auto coord = static_cast<std::uint32_t>(pos / cell_size);
  return coord < limit ? coord : limit - 1U;
}

__device__ bool cell_may_intersect_radius(std::uint32_t cx, std::uint32_t cy, float cell_size, float origin_x,
                                          float origin_y, float radius) {
  const auto center_x = (static_cast<float>(cx) + 0.5F) * cell_size;
  const auto center_y = (static_cast<float>(cy) + 0.5F) * cell_size;
  const auto dx = center_x - origin_x;
  const auto dy = center_y - origin_y;
  const auto half_size = cell_size * 0.5F;
  const auto nearest_x = fmaxf(fabsf(dx) - half_size, 0.0F);
  const auto nearest_y = fmaxf(fabsf(dy) - half_size, 0.0F);
  return (nearest_x * nearest_x) + (nearest_y * nearest_y) <= radius * radius;
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

__global__ void count_population_cells_kernel(DevicePopulationBuffers population, std::uint32_t *cell_counts,
                                              std::uint32_t grid_cols, std::uint32_t grid_rows, float cell_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  const auto cx = cell_coord(population.pos_x[idx], cell_size, grid_cols);
  const auto cy = cell_coord(population.pos_y[idx], cell_size, grid_rows);
  atomicAdd(&cell_counts[(cy * grid_cols) + cx], 1U);
}

__global__ void scatter_population_cells_kernel(DevicePopulationBuffers population, std::uint32_t *cell_write_offsets,
                                                PopulationGridEntry *entries, std::uint32_t grid_cols,
                                                std::uint32_t grid_rows, float cell_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  const auto cx = cell_coord(population.pos_x[idx], cell_size, grid_cols);
  const auto cy = cell_coord(population.pos_y[idx], cell_size, grid_rows);
  const auto slot = atomicAdd(&cell_write_offsets[(cy * grid_cols) + cx], 1U);
  entries[slot] = PopulationGridEntry{idx, population.pos_x[idx], population.pos_y[idx]};
}

__global__ void count_food_cells_kernel(FoodBuffer food, std::uint32_t *cell_counts, std::uint32_t grid_cols,
                                        std::uint32_t grid_rows, float cell_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= food.capacity || food.active[idx] == 0U) {
    return;
  }

  const auto cx = cell_coord(food.pos_x[idx], cell_size, grid_cols);
  const auto cy = cell_coord(food.pos_y[idx], cell_size, grid_rows);
  atomicAdd(&cell_counts[(cy * grid_cols) + cx], 1U);
}

__global__ void scatter_food_cells_kernel(FoodBuffer food, std::uint32_t *cell_write_offsets, FoodGridEntry *entries,
                                          std::uint32_t grid_cols, std::uint32_t grid_rows, float cell_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= food.capacity || food.active[idx] == 0U) {
    return;
  }

  const auto cx = cell_coord(food.pos_x[idx], cell_size, grid_cols);
  const auto cy = cell_coord(food.pos_y[idx], cell_size, grid_rows);
  const auto slot = atomicAdd(&cell_write_offsets[(cy * grid_cols) + cx], 1U);
  entries[slot] = FoodGridEntry{idx, food.pos_x[idx], food.pos_y[idx]};
}

__global__ void build_cell_offsets_kernel(const std::uint32_t *cell_counts, std::uint32_t *cell_offsets,
                                          std::uint32_t *cell_write_offsets, std::uint32_t cell_count) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  std::uint32_t offset = 0U;
  for (std::uint32_t cell = 0; cell < cell_count; ++cell) {
    cell_offsets[cell] = offset;
    cell_write_offsets[cell] = offset;
    offset += cell_counts[cell];
  }
  cell_offsets[cell_count] = offset;
}

__global__ void reset_reproduction_state_kernel(std::uint32_t *mate_claims, std::uint32_t capacity,
                                                PopulationKind population_kind, ReproductionSummaryReadback *out_summary,
                                                std::uint32_t *out_pair_count) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < capacity) {
    mate_claims[idx] = moonai_gpu::kUnclaimedMate;
  }
  if (idx == 0U) {
    *out_summary = ReproductionSummaryReadback{population_kind, 0U, 0U, 0U, 0U};
    *out_pair_count = 0U;
  }
}

__global__ void find_reproduction_pairs_kernel(DevicePopulationBuffers population,
                                               const std::uint32_t *cell_offsets,
                                               const PopulationGridEntry *entries,
                                               std::uint32_t grid_cols, std::uint32_t grid_rows,
                                               float grid_cell_size, float mate_range,
                                               float reproduction_energy_threshold, std::uint32_t *mate_claims,
                                               ReproductionPair *out_pairs, std::uint32_t *out_pair_count,
                                               ReproductionSummaryReadback *out_summary) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }
  if (population.energy[idx] < reproduction_energy_threshold) {
    return;
  }

  atomicAdd(&out_summary->eligible_parents, 1U);
  const auto px = population.pos_x[idx];
  const auto py = population.pos_y[idx];
  const auto mate_range_sq = mate_range * mate_range;
  const auto cells_to_check = static_cast<std::int32_t>(mate_range / grid_cell_size) + 1;
  const auto base_cx = static_cast<std::int32_t>(cell_coord(px, grid_cell_size, grid_cols));
  const auto base_cy = static_cast<std::int32_t>(cell_coord(py, grid_cell_size, grid_rows));

  std::uint32_t best_mate = population.capacity;
  float best_distance_sq = mate_range_sq;
  for (auto dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    const auto cy = base_cy + dy_cell;
    if (cy < 0 || cy >= static_cast<std::int32_t>(grid_rows)) {
      continue;
    }
    for (auto dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const auto cx = base_cx + dx_cell;
      if (cx < 0 || cx >= static_cast<std::int32_t>(grid_cols)) {
        continue;
      }
      if (!cell_may_intersect_radius(static_cast<std::uint32_t>(cx), static_cast<std::uint32_t>(cy), grid_cell_size,
                                     px, py, mate_range)) {
        continue;
      }

      const auto cell = (static_cast<std::uint32_t>(cy) * grid_cols) + static_cast<std::uint32_t>(cx);
      for (auto slot = cell_offsets[cell]; slot < cell_offsets[cell + 1U]; ++slot) {
        const auto entry = entries[slot];
        if (entry.slot <= idx || population.alive[entry.slot] == 0U ||
            population.energy[entry.slot] < reproduction_energy_threshold) {
          continue;
        }

        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto dist_sq = (dx * dx) + (dy * dy);
        if (dist_sq > best_distance_sq || dist_sq <= 0.0F) {
          continue;
        }
        best_distance_sq = dist_sq;
        best_mate = entry.slot;
      }
    }
  }

  if (best_mate == population.capacity) {
    return;
  }

  atomicAdd(&out_summary->candidate_pairs, 1U);
  if (atomicCAS(&mate_claims[best_mate], moonai_gpu::kUnclaimedMate, idx) != moonai_gpu::kUnclaimedMate) {
    return;
  }

  const auto pair_index = atomicAdd(out_pair_count, 1U);
  out_pairs[pair_index] = ReproductionPair{idx, best_mate};
}

__global__ void apply_reproduction_energy_kernel(DevicePopulationBuffers population, const ReproductionPair *pairs,
                                                 std::uint32_t pair_count, float energy_cost,
                                                 std::uint32_t *birth_counter, std::uint32_t *death_counter,
                                                 ReproductionSummaryReadback *out_summary) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= pair_count) {
    return;
  }

  const auto pair = pairs[idx];
  for (const auto slot : {pair.parent_a_slot, pair.parent_b_slot}) {
    population.energy[slot] -= energy_cost;
    if (population.energy[slot] <= 0.0F) {
      population.energy[slot] = 0.0F;
      population.alive[slot] = 0U;
      population.vel_x[slot] = 0.0F;
      population.vel_y[slot] = 0.0F;
      atomicAdd(death_counter, 1U);
    }
  }

  atomicAdd(birth_counter, 1U);
  atomicAdd(&out_summary->births, 1U);
}

template <bool SelfIsPredator>
__global__ void compute_sensor_inputs_kernel(DevicePopulationBuffers self_population,
                                             const std::uint32_t *predator_cell_offsets,
                                             const PopulationGridEntry *predator_entries,
                                             const std::uint32_t *prey_cell_offsets,
                                             const PopulationGridEntry *prey_entries,
                                             const std::uint32_t *food_cell_offsets, const FoodGridEntry *food_entries,
                                             std::uint32_t grid_cols, std::uint32_t grid_rows, float grid_cell_size,
                                             std::uint32_t num_inputs, float vision_range, float max_energy,
                                             float agent_speed, float world_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= self_population.capacity || self_population.sensor_inputs == nullptr) {
    return;
  }

  const auto sensor_base = static_cast<std::size_t>(idx) * num_inputs;
  auto *out = self_population.sensor_inputs + sensor_base;
  for (std::uint32_t sensor_idx = 0; sensor_idx < num_inputs; ++sensor_idx) {
    out[sensor_idx] = 0.0F;
  }

  if (self_population.alive[idx] == 0U) {
    return;
  }

  const auto px = self_population.pos_x[idx];
  const auto py = self_population.pos_y[idx];
  const auto vision_sq = vision_range * vision_range;
  const auto cells_to_check = static_cast<std::int32_t>(vision_range / grid_cell_size) + 1;
  const auto base_cx = static_cast<std::int32_t>(cell_coord(px, grid_cell_size, grid_cols));
  const auto base_cy = static_cast<std::int32_t>(cell_coord(py, grid_cell_size, grid_rows));
  float predator_dx[moonai_gpu::kNearestTargetsPerType];
  float predator_dy[moonai_gpu::kNearestTargetsPerType];
  float predator_dist_sq[moonai_gpu::kNearestTargetsPerType];
  float prey_dx[moonai_gpu::kNearestTargetsPerType];
  float prey_dy[moonai_gpu::kNearestTargetsPerType];
  float prey_dist_sq[moonai_gpu::kNearestTargetsPerType];
  float food_dx[moonai_gpu::kNearestTargetsPerType];
  float food_dy[moonai_gpu::kNearestTargetsPerType];
  float food_dist_sq[moonai_gpu::kNearestTargetsPerType];
  for (std::uint32_t nearest_idx = 0; nearest_idx < moonai_gpu::kNearestTargetsPerType; ++nearest_idx) {
    predator_dx[nearest_idx] = 0.0F;
    predator_dy[nearest_idx] = 0.0F;
    predator_dist_sq[nearest_idx] = INFINITY;
    prey_dx[nearest_idx] = 0.0F;
    prey_dy[nearest_idx] = 0.0F;
    prey_dist_sq[nearest_idx] = INFINITY;
    food_dx[nearest_idx] = 0.0F;
    food_dy[nearest_idx] = 0.0F;
    food_dist_sq[nearest_idx] = INFINITY;
  }

  for (auto dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    const auto cy = base_cy + dy_cell;
    if (cy < 0 || cy >= static_cast<std::int32_t>(grid_rows)) {
      continue;
    }
    for (auto dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const auto cx = base_cx + dx_cell;
      if (cx < 0 || cx >= static_cast<std::int32_t>(grid_cols)) {
        continue;
      }
      if (!cell_may_intersect_radius(static_cast<std::uint32_t>(cx), static_cast<std::uint32_t>(cy), grid_cell_size,
                                     px, py, vision_range)) {
        continue;
      }

      const auto cell = (static_cast<std::uint32_t>(cy) * grid_cols) + static_cast<std::uint32_t>(cx);
      for (auto slot = predator_cell_offsets[cell]; slot < predator_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = predator_entries[slot];
        if (SelfIsPredator && entry.slot == idx) {
          continue;
        }
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto dist_sq = (dx * dx) + (dy * dy);
        if (dist_sq > vision_sq || dist_sq <= 0.0F) {
          continue;
        }
        insert_nearest_candidate(dx, dy, dist_sq, predator_dx, predator_dy, predator_dist_sq);
      }

      for (auto slot = prey_cell_offsets[cell]; slot < prey_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = prey_entries[slot];
        if (!SelfIsPredator && entry.slot == idx) {
          continue;
        }
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto dist_sq = (dx * dx) + (dy * dy);
        if (dist_sq > vision_sq || dist_sq <= 0.0F) {
          continue;
        }
        insert_nearest_candidate(dx, dy, dist_sq, prey_dx, prey_dy, prey_dist_sq);
      }

      for (auto slot = food_cell_offsets[cell]; slot < food_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = food_entries[slot];
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto dist_sq = (dx * dx) + (dy * dy);
        if (dist_sq > vision_sq || dist_sq <= 0.0F) {
          continue;
        }
        insert_nearest_candidate(dx, dy, dist_sq, food_dx, food_dy, food_dist_sq);
      }
    }
  }

  if (num_inputs > 0U) {
    encode_nearest_targets(predator_dx, predator_dy, predator_dist_sq, vision_range, out,
                           num_inputs < moonai_gpu::kPerTypeSensorCount ? num_inputs : moonai_gpu::kPerTypeSensorCount);
  }
  if (num_inputs > moonai_gpu::kPerTypeSensorCount) {
    const auto prey_output_capacity = num_inputs - moonai_gpu::kPerTypeSensorCount;
    encode_nearest_targets(prey_dx, prey_dy, prey_dist_sq, vision_range, out + moonai_gpu::kPerTypeSensorCount,
                           prey_output_capacity < moonai_gpu::kPerTypeSensorCount ? prey_output_capacity
                                                                                   : moonai_gpu::kPerTypeSensorCount);
  }
  if (num_inputs > (2U * moonai_gpu::kPerTypeSensorCount)) {
    const auto food_output_capacity = num_inputs - (2U * moonai_gpu::kPerTypeSensorCount);
    encode_nearest_targets(food_dx, food_dy, food_dist_sq, vision_range, out + (2U * moonai_gpu::kPerTypeSensorCount),
                           food_output_capacity < moonai_gpu::kPerTypeSensorCount ? food_output_capacity
                                                                                   : moonai_gpu::kPerTypeSensorCount);
  }
  if (num_inputs > moonai_gpu::kSelfEnergyInputIndex) {
    out[moonai_gpu::kSelfEnergyInputIndex] = max_energy <= 0.0F ? 0.0F
                                                                 : clamp_range(self_population.energy[idx] / max_energy, 0.0F, 1.0F);
  }
  if (agent_speed > 0.0F && num_inputs > moonai_gpu::kVelocityXInputIndex) {
    out[moonai_gpu::kVelocityXInputIndex] = clamp_range(self_population.vel_x[idx] / agent_speed, -1.0F, 1.0F);
  }
  if (agent_speed > 0.0F && num_inputs > moonai_gpu::kVelocityYInputIndex) {
    out[moonai_gpu::kVelocityYInputIndex] = clamp_range(self_population.vel_y[idx] / agent_speed, -1.0F, 1.0F);
  }
  if (num_inputs > moonai_gpu::kWallXInputIndex) {
    out[moonai_gpu::kWallXInputIndex] = encode_axis_wall_sensor(px, world_size - px, vision_range);
  }
  if (num_inputs > moonai_gpu::kWallYInputIndex) {
    out[moonai_gpu::kWallYInputIndex] = encode_axis_wall_sensor(py, world_size - py, vision_range);
  }
}

__global__ void reset_simulation_counters_kernel(SimulationCounters *counters) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *counters = SimulationCounters{};
  }
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

__global__ void sensor_snapshot_kernel(DevicePopulationBuffers population, PopulationKind population_kind,
                                       std::uint32_t slot, std::uint32_t num_inputs,
                                       SensorSnapshotReadback *out_snapshot) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  out_snapshot->population_kind = population_kind;
  out_snapshot->slot = slot;
  out_snapshot->input_count = static_cast<std::uint16_t>(
      num_inputs < moonai_gpu::kSensorInputCount ? num_inputs : moonai_gpu::kSensorInputCount);
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

__global__ void update_vitals_kernel(DevicePopulationBuffers population, float energy_drain_per_tick, std::uint32_t max_age,
                                     std::uint32_t *free_list, std::uint32_t *free_len, std::uint32_t *death_counter) {
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

__global__ void resolve_food_kernel(DevicePopulationBuffers prey, FoodBuffer food, SimulationCounters *counters,
                                    float interaction_range, float energy_gain_from_food, float max_energy,
                                    float world_size) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  const auto interaction_range_sq = interaction_range * interaction_range;
  for (std::uint32_t prey_idx = 0; prey_idx < prey.capacity; ++prey_idx) {
    if (prey.alive[prey_idx] == 0U) {
      continue;
    }

    std::uint32_t best_food = food.capacity;
    float best_distance = interaction_range_sq;
    for (std::uint32_t food_idx = 0; food_idx < food.capacity; ++food_idx) {
      if (food.active[food_idx] == 0U) {
        continue;
      }

      const auto dx = food.pos_x[food_idx] - prey.pos_x[prey_idx];
      const auto dy = food.pos_y[food_idx] - prey.pos_y[prey_idx];
      const auto distance_sq = (dx * dx) + (dy * dy);
      if (distance_sq <= best_distance) {
        best_distance = distance_sq;
        best_food = food_idx;
      }
    }

    if (best_food == food.capacity) {
      continue;
    }

    auto rng = prey.rng_state[prey_idx] ^ (static_cast<std::uint64_t>(best_food) << 16U) ^ counters->tick;
    food.pos_x[best_food] = moonai_gpu::next_unit_float(rng) * world_size;
    food.pos_y[best_food] = moonai_gpu::next_unit_float(rng) * world_size;
    food.active[best_food] = 1U;
    prey.rng_state[prey_idx] = rng;
    prey.energy[prey_idx] += energy_gain_from_food;
    if (prey.energy[prey_idx] > max_energy) {
      prey.energy[prey_idx] = max_energy;
    }
    ++counters->food_eaten;
  }
}

__global__ void resolve_combat_kernel(DevicePopulationBuffers predator, DevicePopulationBuffers prey,
                                      SimulationCounters *counters, std::uint32_t *prey_free_list,
                                      std::uint32_t *prey_free_len, float interaction_range, float energy_gain_from_kill,
                                      float max_energy) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  const auto interaction_range_sq = interaction_range * interaction_range;
  for (std::uint32_t predator_idx = 0; predator_idx < predator.capacity; ++predator_idx) {
    if (predator.alive[predator_idx] == 0U) {
      continue;
    }

    std::uint32_t best_prey = prey.capacity;
    float best_distance = interaction_range_sq;
    for (std::uint32_t prey_idx = 0; prey_idx < prey.capacity; ++prey_idx) {
      if (prey.alive[prey_idx] == 0U) {
        continue;
      }

      const auto dx = prey.pos_x[prey_idx] - predator.pos_x[predator_idx];
      const auto dy = prey.pos_y[prey_idx] - predator.pos_y[predator_idx];
      const auto distance_sq = (dx * dx) + (dy * dy);
      if (distance_sq <= best_distance) {
        best_distance = distance_sq;
        best_prey = prey_idx;
      }
    }

    if (best_prey == prey.capacity) {
      continue;
    }

    prey.alive[best_prey] = 0U;
    prey.energy[best_prey] = 0.0F;
    prey.vel_x[best_prey] = 0.0F;
    prey.vel_y[best_prey] = 0.0F;
    const auto free_index = (*prey_free_len)++;
    if (free_index < prey.capacity) {
      prey_free_list[free_index] = best_prey;
    }
    predator.energy[predator_idx] += energy_gain_from_kill;
    if (predator.energy[predator_idx] > max_energy) {
      predator.energy[predator_idx] = max_energy;
    }
    ++counters->kills;
    ++counters->prey_deaths;
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

__global__ void metrics_reduce_kernel(DevicePopulationBuffers predator, DevicePopulationBuffers prey,
                                      const SimulationCounters *counters, MetricsSummaryReadback *out_metrics) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  bool predator_species_seen[moonai_gpu::kSpeciesBucketCount]{};
  bool prey_species_seen[moonai_gpu::kSpeciesBucketCount]{};
  std::uint32_t predator_count = 0U;
  std::uint32_t prey_count = 0U;
  float predator_energy_sum = 0.0F;
  float prey_energy_sum = 0.0F;
  float predator_complexity_sum = 0.0F;
  float prey_complexity_sum = 0.0F;
  std::uint32_t predator_generation_sum = 0U;
  std::uint32_t prey_generation_sum = 0U;
  std::uint32_t max_predator_generation = 0U;
  std::uint32_t max_prey_generation = 0U;
  std::uint32_t predator_species = 0U;
  std::uint32_t prey_species = 0U;

  for (std::uint32_t idx = 0; idx < predator.capacity; ++idx) {
    if (predator.alive[idx] == 0U) {
      continue;
    }
    ++predator_count;
    predator_energy_sum += predator.energy[idx];
    predator_complexity_sum += static_cast<float>(predator.genome.num_nodes[idx] +
                                                  moonai_gpu::count_enabled_connections(
                                                      predator, idx, predator.genome.num_connections[idx]));
    predator_generation_sum += predator.generation[idx];
    if (predator.generation[idx] > max_predator_generation) {
      max_predator_generation = predator.generation[idx];
    }
    if (predator.species_id[idx] < moonai_gpu::kSpeciesBucketCount && !predator_species_seen[predator.species_id[idx]]) {
      predator_species_seen[predator.species_id[idx]] = true;
      ++predator_species;
    }
  }

  for (std::uint32_t idx = 0; idx < prey.capacity; ++idx) {
    if (prey.alive[idx] == 0U) {
      continue;
    }
    ++prey_count;
    prey_energy_sum += prey.energy[idx];
    prey_complexity_sum += static_cast<float>(prey.genome.num_nodes[idx] +
                                              moonai_gpu::count_enabled_connections(prey, idx,
                                                                                    prey.genome.num_connections[idx]));
    prey_generation_sum += prey.generation[idx];
    if (prey.generation[idx] > max_prey_generation) {
      max_prey_generation = prey.generation[idx];
    }
    if (prey.species_id[idx] < moonai_gpu::kSpeciesBucketCount && !prey_species_seen[prey.species_id[idx]]) {
      prey_species_seen[prey.species_id[idx]] = true;
      ++prey_species;
    }
  }

  out_metrics->tick = counters->tick;
  out_metrics->predator_count = predator_count;
  out_metrics->prey_count = prey_count;
  out_metrics->predator_births = counters->predator_births;
  out_metrics->prey_births = counters->prey_births;
  out_metrics->predator_deaths = counters->predator_deaths;
  out_metrics->prey_deaths = counters->prey_deaths;
  out_metrics->predator_species = predator_species;
  out_metrics->prey_species = prey_species;
  out_metrics->avg_predator_complexity = predator_count == 0U ? 0.0F : predator_complexity_sum / static_cast<float>(predator_count);
  out_metrics->avg_prey_complexity = prey_count == 0U ? 0.0F : prey_complexity_sum / static_cast<float>(prey_count);
  out_metrics->avg_predator_energy = predator_count == 0U ? 0.0F : predator_energy_sum / static_cast<float>(predator_count);
  out_metrics->avg_prey_energy = prey_count == 0U ? 0.0F : prey_energy_sum / static_cast<float>(prey_count);
  out_metrics->max_predator_generation = max_predator_generation;
  out_metrics->avg_predator_generation =
      predator_count == 0U ? 0.0F : static_cast<float>(predator_generation_sum) / static_cast<float>(predator_count);
  out_metrics->max_prey_generation = max_prey_generation;
  out_metrics->avg_prey_generation = prey_count == 0U ? 0.0F : static_cast<float>(prey_generation_sum) / static_cast<float>(prey_count);
}

__global__ void write_ui_stats_kernel(DevicePopulationBuffers predator, DevicePopulationBuffers prey, FoodBuffer food,
                                      const SimulationCounters *counters, UiStatsReadback *out_stats) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  std::uint32_t predator_count = 0U;
  std::uint32_t prey_count = 0U;
  float predator_energy = 0.0F;
  float prey_energy = 0.0F;
  for (std::uint32_t idx = 0; idx < predator.capacity; ++idx) {
    if (predator.alive[idx] == 0U) {
      continue;
    }
    ++predator_count;
    predator_energy += predator.energy[idx];
  }
  for (std::uint32_t idx = 0; idx < prey.capacity; ++idx) {
    if (prey.alive[idx] == 0U) {
      continue;
    }
    ++prey_count;
    prey_energy += prey.energy[idx];
  }
  out_stats->tick = counters->tick;
  out_stats->predator_count = predator_count;
  out_stats->prey_count = prey_count;
  out_stats->predator_births = counters->predator_births;
  out_stats->prey_births = counters->prey_births;
  out_stats->predator_deaths = counters->predator_deaths;
  out_stats->prey_deaths = counters->prey_deaths;
  out_stats->kills = counters->kills;
  out_stats->food_eaten = counters->food_eaten;
  out_stats->avg_predator_energy = predator_count == 0U ? 0.0F : predator_energy / static_cast<float>(predator_count);
  out_stats->avg_prey_energy = prey_count == 0U ? 0.0F : prey_energy / static_cast<float>(prey_count);
}

__global__ void free_list_state_kernel(FoodBuffer food, const SimulationCounters *counters,
                                       const std::uint32_t *predator_free_len, const std::uint32_t *prey_free_len,
                                       moonai_gpu::FreeListStateReadback *out_state) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  std::uint32_t active_food = 0U;
  for (std::uint32_t idx = 0; idx < food.capacity; ++idx) {
    active_food += food.active[idx] != 0U;
  }
  out_state->tick = counters->tick;
  out_state->predator_free_slots = *predator_free_len;
  out_state->prey_free_slots = *prey_free_len;
  out_state->active_food_count = active_food;
  out_state->food_capacity = food.capacity;
}

__global__ void render_snapshot_kernel(DevicePopulationBuffers predator, DevicePopulationBuffers prey, FoodBuffer food,
                                       const SimulationCounters *counters, std::uint32_t max_predators,
                                       std::uint32_t max_prey, std::uint32_t max_food, RenderSnapshotHeader *out_header,
                                       RenderAgentReadback *out_predators, RenderAgentReadback *out_prey,
                                       RenderFoodReadback *out_food) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  out_header->tick = counters->tick;
  out_header->total_predators = 0U;
  out_header->total_prey = 0U;
  out_header->total_food = 0U;
  out_header->returned_predators = 0U;
  out_header->returned_prey = 0U;
  out_header->returned_food = 0U;

  for (std::uint32_t idx = 0; idx < predator.capacity; ++idx) {
    if (predator.alive[idx] == 0U) {
      continue;
    }
    ++out_header->total_predators;
    if (out_header->returned_predators < max_predators) {
      out_predators[out_header->returned_predators++] = RenderAgentReadback{PopulationKind::Predator,
                                                                            idx,
                                                                            predator.entity_id[idx],
                                                                            predator.species_id[idx],
                                                                            predator.generation[idx],
                                                                            predator.age[idx],
                                                                            predator.pos_x[idx],
                                                                            predator.pos_y[idx],
                                                                            predator.vel_x[idx],
                                                                            predator.vel_y[idx],
                                                                            predator.energy[idx]};
    }
  }
  for (std::uint32_t idx = 0; idx < prey.capacity; ++idx) {
    if (prey.alive[idx] == 0U) {
      continue;
    }
    ++out_header->total_prey;
    if (out_header->returned_prey < max_prey) {
      out_prey[out_header->returned_prey++] = RenderAgentReadback{PopulationKind::Prey,
                                                                  idx,
                                                                  prey.entity_id[idx],
                                                                  prey.species_id[idx],
                                                                  prey.generation[idx],
                                                                  prey.age[idx],
                                                                  prey.pos_x[idx],
                                                                  prey.pos_y[idx],
                                                                  prey.vel_x[idx],
                                                                  prey.vel_y[idx],
                                                                  prey.energy[idx]};
    }
  }
  for (std::uint32_t idx = 0; idx < food.capacity; ++idx) {
    if (food.active[idx] == 0U) {
      continue;
    }
    ++out_header->total_food;
    if (out_header->returned_food < max_food) {
      out_food[out_header->returned_food++] = RenderFoodReadback{idx, food.active[idx], 0U, 0U, food.pos_x[idx],
                                                                 food.pos_y[idx]};
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

CudaStatus build_spatial_grid(GpuEvolutionState &state) {
  const auto cell_count = state.grid_cols * state.grid_rows;
  if (cell_count == 0U) {
    return CudaStatus::InvalidArgument;
  }

  auto status = moonai_gpu::zero_device_memory(state.predator_cell_counts, sizeof(std::uint32_t) * cell_count);
  if (status != CudaStatus::Success) {
    return status;
  }
  status = moonai_gpu::zero_device_memory(state.prey_cell_counts, sizeof(std::uint32_t) * cell_count);
  if (status != CudaStatus::Success) {
    return status;
  }
  status = moonai_gpu::zero_device_memory(state.food_cell_counts, sizeof(std::uint32_t) * cell_count);
  if (status != CudaStatus::Success) {
    return status;
  }

  const auto predator_blocks = (state.predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state.prey.capacity + 255U) / 256U;
  const auto food_blocks = (state.food.capacity + 255U) / 256U;
  count_population_cells_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state.predator, state.predator_cell_counts, state.grid_cols, state.grid_rows, state.grid_cell_size);
  count_population_cells_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state.prey, state.prey_cell_counts, state.grid_cols, state.grid_rows, state.grid_cell_size);
  count_food_cells_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(state.food, state.food_cell_counts,
                                                                            state.grid_cols, state.grid_rows,
                                                                            state.grid_cell_size);
  build_cell_offsets_kernel<<<1U, 1U>>>(state.predator_cell_counts, state.predator_cell_offsets,
                                        state.predator_cell_write_offsets, cell_count);
  build_cell_offsets_kernel<<<1U, 1U>>>(state.prey_cell_counts, state.prey_cell_offsets, state.prey_cell_write_offsets,
                                        cell_count);
  build_cell_offsets_kernel<<<1U, 1U>>>(state.food_cell_counts, state.food_cell_offsets, state.food_cell_write_offsets,
                                        cell_count);
  scatter_population_cells_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state.predator, state.predator_cell_write_offsets, state.predator_grid_entries, state.grid_cols, state.grid_rows,
      state.grid_cell_size);
  scatter_population_cells_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state.prey, state.prey_cell_write_offsets, state.prey_grid_entries, state.grid_cols, state.grid_rows,
      state.grid_cell_size);
  scatter_food_cells_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(
      state.food, state.food_cell_write_offsets, state.food_grid_entries, state.grid_cols, state.grid_rows,
      state.grid_cell_size);
  return CudaStatus::Success;
}

CudaStatus read_device_u32(const std::uint32_t *device_ptr, std::uint32_t &host_value) {
  return moonai_gpu::copy_compact_device_readback(device_ptr, &host_value, sizeof(host_value));
}

CudaStatus read_reproduction_pairs(const ReproductionPair *device_pairs, std::uint32_t pair_count,
                                   std::vector<ReproductionPair> &pairs) {
  pairs.assign(pair_count, ReproductionPair{});
  if (pair_count == 0U) {
    return CudaStatus::Success;
  }
  return moonai_gpu::copy_compact_device_readback(device_pairs, pairs.data(), sizeof(ReproductionPair) * pair_count);
}

CudaStatus read_free_slots(const GpuEvolutionState &state, PopulationKind population_kind, std::vector<std::uint32_t> &free_slots) {
  const auto *device_free_len = population_kind == PopulationKind::Predator ? state.predator_free_len : state.prey_free_len;
  const auto *device_free_list = population_kind == PopulationKind::Predator ? state.predator_free_list : state.prey_free_list;
  std::uint32_t free_len = 0U;
  auto status = read_device_u32(device_free_len, free_len);
  if (status != CudaStatus::Success) {
    return status;
  }

  free_slots.assign(free_len, 0U);
  if (free_len == 0U) {
    return CudaStatus::Success;
  }
  return moonai_gpu::copy_compact_device_readback(device_free_list, free_slots.data(), sizeof(std::uint32_t) * free_len);
}

CudaStatus ensure_birth_capacity(GpuEvolutionState &state, PopulationKind population_kind, std::uint32_t births_pending) {
  if (births_pending == 0U) {
    return CudaStatus::Success;
  }

  PopulationSummaryReadback summary{};
  auto status = static_cast<CudaStatus>(moonai_gpu_evolution_population_summary(&state, population_kind, &summary));
  if (status != CudaStatus::Success) {
    return status;
  }

  std::uint32_t free_slots = 0U;
  status = read_device_u32(population_kind == PopulationKind::Predator ? state.predator_free_len : state.prey_free_len, free_slots);
  if (status != CudaStatus::Success) {
    return status;
  }

  auto capacity = summary.capacity;
  const auto required_live = summary.live_count + births_pending;
  if (free_slots >= births_pending && required_live <= ((capacity * 9U) / 10U)) {
    return CudaStatus::Success;
  }

  std::uint32_t new_capacity = capacity == 0U ? 1U : capacity;
  while ((new_capacity - summary.live_count) < births_pending || required_live > ((new_capacity * 9U) / 10U)) {
    new_capacity = new_capacity == 0U ? 1U : new_capacity * 2U;
  }
  return expand_population_capacity(state, population_kind, new_capacity);
}

CudaStatus refresh_metrics_summary(GpuEvolutionState &state) {
  moonai_gpu::SpeciesBatchReadbackHeader predator_header{};
  moonai_gpu::SpeciesBatchReadbackHeader prey_header{};
  auto status = static_cast<CudaStatus>(
      moonai_gpu_evolution_species_summaries(&state, PopulationKind::Predator, 0U, &predator_header, nullptr, nullptr));
  if (status != CudaStatus::Success) {
    return status;
  }
  status = static_cast<CudaStatus>(
      moonai_gpu_evolution_species_summaries(&state, PopulationKind::Prey, 0U, &prey_header, nullptr, nullptr));
  if (status != CudaStatus::Success) {
    return status;
  }
  metrics_reduce_kernel<<<1U, 1U>>>(state.predator, state.prey, state.counters, state.metrics_summary);
  return moonai_gpu::synchronize_kernels();
}

CudaStatus run_reproduction_for_population(GpuEvolutionState &state, PopulationKind population_kind) {
  auto &population = moonai_gpu::population_for_kind(state, population_kind);
  auto *mate_claims = population_kind == PopulationKind::Predator ? state.predator_mate_claims : state.prey_mate_claims;
  auto *pair_buffer = population_kind == PopulationKind::Predator ? state.predator_reproduction_pairs : state.prey_reproduction_pairs;
  auto *pair_count_ptr = population_kind == PopulationKind::Predator ? state.predator_pair_count : state.prey_pair_count;
  auto *summary_ptr =
      population_kind == PopulationKind::Predator ? state.predator_reproduction_summary : state.prey_reproduction_summary;
  auto *cell_offsets = population_kind == PopulationKind::Predator ? state.predator_cell_offsets : state.prey_cell_offsets;
  auto *entries = population_kind == PopulationKind::Predator ? state.predator_grid_entries : state.prey_grid_entries;
  auto *birth_counter = population_kind == PopulationKind::Predator ? &state.counters->predator_births : &state.counters->prey_births;
  auto *death_counter = population_kind == PopulationKind::Predator ? &state.counters->predator_deaths : &state.counters->prey_deaths;

  const auto blocks = (population.capacity + 255U) / 256U;
  reset_reproduction_state_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(mate_claims, population.capacity, population_kind,
                                                                         summary_ptr, pair_count_ptr);
  find_reproduction_pairs_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(
      population, cell_offsets, entries, state.grid_cols, state.grid_rows, state.grid_cell_size, state.simulation.mate_range,
      state.simulation.reproduction_energy_threshold, mate_claims, pair_buffer, pair_count_ptr, summary_ptr);
  auto status = moonai_gpu::synchronize_kernels();
  if (status != CudaStatus::Success) {
    return status;
  }

  std::uint32_t pair_count = 0U;
  status = read_device_u32(pair_count_ptr, pair_count);
  if (status != CudaStatus::Success || pair_count == 0U) {
    return status;
  }

  std::vector<ReproductionPair> pairs;
  status = read_reproduction_pairs(pair_buffer, pair_count, pairs);
  if (status != CudaStatus::Success) {
    return status;
  }
  ReproductionSummaryReadback summary{};
  status = moonai_gpu::copy_compact_device_readback(summary_ptr, &summary, sizeof(summary));
  if (status != CudaStatus::Success) {
    return status;
  }

  status = ensure_birth_capacity(state, population_kind, pair_count);
  if (status != CudaStatus::Success) {
    return status;
  }

  pair_buffer = population_kind == PopulationKind::Predator ? state.predator_reproduction_pairs : state.prey_reproduction_pairs;
  pair_count_ptr = population_kind == PopulationKind::Predator ? state.predator_pair_count : state.prey_pair_count;
  summary_ptr = population_kind == PopulationKind::Predator ? state.predator_reproduction_summary : state.prey_reproduction_summary;
  status = moonai_gpu::copy_host_data_to_device(summary_ptr, &summary, sizeof(summary));
  if (status != CudaStatus::Success) {
    return status;
  }
  status = moonai_gpu::copy_host_data_to_device(pair_count_ptr, &pair_count, sizeof(pair_count));
  if (status != CudaStatus::Success) {
    return status;
  }
  status = moonai_gpu::copy_host_data_to_device(pair_buffer, pairs.data(), sizeof(ReproductionPair) * pair_count);
  if (status != CudaStatus::Success) {
    return status;
  }

  std::vector<std::uint32_t> free_slots;
  status = read_free_slots(state, population_kind, free_slots);
  if (status != CudaStatus::Success) {
    return status;
  }

  const GpuMutationConfig mutation_config{state.simulation.mutation_rate,
                                          state.simulation.weight_mutation_power,
                                          state.simulation.add_node_rate,
                                          state.simulation.add_connection_rate,
                                          state.simulation.delete_connection_rate,
                                          state.simulation.max_connection_attempts};
  std::uint32_t births_applied = 0U;
  for (std::uint32_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    if (pair_index >= free_slots.size()) {
      break;
    }
    const auto offspring_slot = free_slots[pair_index];
    moonai_gpu::CrossoverSummaryReadback crossover_summary{};
    status = static_cast<CudaStatus>(moonai_gpu_evolution_crossover(&state, population_kind, pairs[pair_index].parent_a_slot,
                                                                    pairs[pair_index].parent_b_slot, offspring_slot,
                                                                    &crossover_summary));
    if (status != CudaStatus::Success) {
      return status;
    }
    moonai_gpu::MutationSummaryReadback mutation_summary{};
    status = static_cast<CudaStatus>(
        moonai_gpu_evolution_mutate_slot(&state, population_kind, offspring_slot, &mutation_config, &mutation_summary));
    if (status != CudaStatus::Success) {
      return status;
    }
    moonai_gpu::CompiledNetworkReadbackHeader compile_header{};
    status = static_cast<CudaStatus>(moonai_gpu_evolution_compile_slot(&state, population_kind, offspring_slot, &compile_header));
    if (status != CudaStatus::Success) {
      return status;
    }
    ++births_applied;
  }

  if (births_applied < pair_count) {
    status = moonai_gpu::copy_compact_device_readback(summary_ptr, &summary, sizeof(summary));
    if (status != CudaStatus::Success) {
      return status;
    }
    summary.failed_pairs += pair_count - births_applied;
    status = moonai_gpu::copy_host_data_to_device(summary_ptr, &summary, sizeof(summary));
    if (status != CudaStatus::Success) {
      return status;
    }
    pair_count = births_applied;
  }

  if (births_applied > 0U) {
    const auto apply_blocks = (births_applied + 255U) / 256U;
    apply_reproduction_energy_kernel<<<apply_blocks == 0U ? 1U : apply_blocks, 256U>>>(
        population, pair_buffer, births_applied, state.simulation.reproduction_energy_cost, birth_counter, death_counter,
        summary_ptr);
    status = moonai_gpu::synchronize_kernels();
    if (status != CudaStatus::Success) {
      return status;
    }
  }

  auto *free_list = population_kind == PopulationKind::Predator ? state.predator_free_list : state.prey_free_list;
  auto *free_len = population_kind == PopulationKind::Predator ? state.predator_free_len : state.prey_free_len;
  initialize_free_list_kernel<<<1U, 1U>>>(population, free_list, free_len);
  return moonai_gpu::synchronize_kernels();
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
           allocate_population_buffers(state->predator, config->predator_capacity, config->num_inputs, config->node_stride,
                                       config->connection_stride, config->num_outputs),
           allocate_population_buffers(state->prey, config->prey_capacity, config->num_inputs, config->node_stride,
                                       config->connection_stride, config->num_outputs),
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

  const auto status = moonai_gpu::launch_single_value_readback(out_summary, [&](PopulationSummaryReadback *device_summary) {
    summarize_population_kernel<<<1U, 1U>>>(moonai_gpu::population_for_kind(*state, population_kind), state->innovation,
                                            state->next_entity_id, population_kind, device_summary);
    return CudaStatus::Success;
  });
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_initialize(void *state_ptr, const GpuSimulationConfig *config) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || config == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  state->simulation = *config;
  if (state->food.capacity != config->food_capacity) {
    free_food_buffers(state->food);
  }
  if (config->food_capacity > 0U && state->food.capacity == 0U) {
    auto status = allocate_food_buffers(state->food, config->food_capacity);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  if (state->counters == nullptr) {
    auto status = moonai_gpu::alloc_array(&state->counters, 1U);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  if (state->predator_free_list == nullptr) {
    auto status = moonai_gpu::alloc_array(&state->predator_free_list, state->predator.capacity);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  if (state->prey_free_list == nullptr) {
    auto status = moonai_gpu::alloc_array(&state->prey_free_list, state->prey.capacity);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  if (state->predator_free_len == nullptr) {
    auto status = moonai_gpu::alloc_array(&state->predator_free_len, 1U);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  if (state->prey_free_len == nullptr) {
    auto status = moonai_gpu::alloc_array(&state->prey_free_len, 1U);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  if (state->mapped_ui_stats_host == nullptr || state->mapped_ui_stats_device == nullptr) {
    auto status = allocate_mapped_ui_stats(&state->mapped_ui_stats_host, &state->mapped_ui_stats_device);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  auto status = ensure_reproduction_buffers(*state);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  status = ensure_metrics_buffer(*state);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  state->grid_cell_size = fmaxf(config->vision_range, 1.0F);
  const auto computed_grid_cols = static_cast<std::uint32_t>(std::ceil(config->world_size / state->grid_cell_size));
  const auto computed_grid_rows = static_cast<std::uint32_t>(std::ceil(config->world_size / state->grid_cell_size));
  state->grid_cols = computed_grid_cols == 0U ? 1U : computed_grid_cols;
  state->grid_rows = computed_grid_rows == 0U ? 1U : computed_grid_rows;
  const auto grid_cell_count = state->grid_cols * state->grid_rows;
  status = ensure_spatial_grid_buffers(*state, grid_cell_count);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  reset_simulation_counters_kernel<<<1U, 1U>>>(state->counters);
  initialize_free_list_kernel<<<1U, 1U>>>(state->predator, state->predator_free_list, state->predator_free_len);
  initialize_free_list_kernel<<<1U, 1U>>>(state->prey, state->prey_free_list, state->prey_free_len);
  if (state->food.capacity > 0U) {
    const auto food_blocks = (state->food.capacity + 255U) / 256U;
    seed_food_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(state->food, config->seed ^ 0xC0FFEEULL,
                                                                      config->world_size);
  }
  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  reset_reproduction_state_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator_mate_claims, state->predator.capacity, PopulationKind::Predator, state->predator_reproduction_summary,
      state->predator_pair_count);
  reset_reproduction_state_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey_mate_claims, state->prey.capacity, PopulationKind::Prey, state->prey_reproduction_summary,
      state->prey_pair_count);
  status = build_spatial_grid(*state);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  compute_sensor_inputs_kernel<true><<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->config.num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.predator_speed, state->simulation.world_size);
  compute_sensor_inputs_kernel<false><<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->config.num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.prey_speed, state->simulation.world_size);
  write_ui_stats_kernel<<<1U, 1U>>>(state->predator, state->prey, state->food, state->counters, state->mapped_ui_stats_device);
  status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    status = refresh_metrics_summary(*state);
  }
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_step(void *state_ptr, UiStatsReadback *out_stats) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_stats == nullptr || state->counters == nullptr || state->mapped_ui_stats_host == nullptr ||
      state->mapped_ui_stats_device == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  auto status = build_spatial_grid(*state);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  compute_sensor_inputs_kernel<true><<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->config.num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.predator_speed, state->simulation.world_size);
  compute_sensor_inputs_kernel<false><<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->config.num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.prey_speed, state->simulation.world_size);
  infer_population_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(state->predator, state->config.num_inputs);
  infer_population_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(state->prey, state->config.num_inputs);
  update_vitals_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(state->predator,
                                                                                 state->simulation.energy_drain_per_tick,
                                                                                 state->simulation.max_age,
                                                                                 state->predator_free_list,
                                                                                 state->predator_free_len,
                                                                                 &state->counters->predator_deaths);
  update_vitals_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(state->prey, state->simulation.energy_drain_per_tick,
                                                                        state->simulation.max_age, state->prey_free_list,
                                                                        state->prey_free_len, &state->counters->prey_deaths);
  resolve_food_kernel<<<1U, 1U>>>(state->prey, state->food, state->counters, state->simulation.interaction_range,
                                  state->simulation.energy_gain_from_food, state->simulation.max_energy,
                                  state->simulation.world_size);
  resolve_combat_kernel<<<1U, 1U>>>(state->predator, state->prey, state->counters, state->prey_free_list,
                                    state->prey_free_len, state->simulation.interaction_range,
                                    state->simulation.energy_gain_from_kill, state->simulation.max_energy);
  apply_movement_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(state->predator,
                                                                                 state->simulation.predator_speed,
                                                                                 state->simulation.world_size);
  apply_movement_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(state->prey, state->simulation.prey_speed,
                                                                         state->simulation.world_size);
  status = build_spatial_grid(*state);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  status = run_reproduction_for_population(*state, PopulationKind::Predator);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  status = build_spatial_grid(*state);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  status = run_reproduction_for_population(*state, PopulationKind::Prey);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  advance_tick_kernel<<<1U, 1U>>>(state->counters);
  status = build_spatial_grid(*state);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  compute_sensor_inputs_kernel<true><<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->config.num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.predator_speed, state->simulation.world_size);
  compute_sensor_inputs_kernel<false><<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->config.num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.prey_speed, state->simulation.world_size);
  if (state->simulation.report_interval_ticks > 0U) {
    MetricsSummaryReadback metrics_summary{};
    auto read_status = moonai_gpu::copy_compact_device_readback(state->counters, &metrics_summary.tick, sizeof(std::uint32_t));
    if (read_status != CudaStatus::Success) {
      return static_cast<std::int32_t>(read_status);
    }
    if (metrics_summary.tick % state->simulation.report_interval_ticks == 0U) {
      status = refresh_metrics_summary(*state);
      if (status != CudaStatus::Success) {
        return static_cast<std::int32_t>(status);
      }
    }
  }
  write_ui_stats_kernel<<<1U, 1U>>>(state->predator, state->prey, state->food, state->counters, state->mapped_ui_stats_device);
  status = moonai_gpu::synchronize_kernels();
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  *out_stats = *state->mapped_ui_stats_host;
  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" std::int32_t moonai_gpu_simulation_ui_stats(const void *state_ptr, UiStatsReadback *out_stats) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_stats == nullptr || state->mapped_ui_stats_host == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  *out_stats = *state->mapped_ui_stats_host;
  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" std::int32_t moonai_gpu_simulation_free_list_state(const void *state_ptr,
                                                                 moonai_gpu::FreeListStateReadback *out_state) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_state == nullptr || state->counters == nullptr || state->predator_free_len == nullptr ||
      state->prey_free_len == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto status = moonai_gpu::launch_single_value_readback(out_state, [&](moonai_gpu::FreeListStateReadback *device_state) {
    free_list_state_kernel<<<1U, 1U>>>(state->food, state->counters, state->predator_free_len, state->prey_free_len,
                                       device_state);
    return CudaStatus::Success;
  });
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_metrics_summary(const void *state_ptr,
                                                                   MetricsSummaryReadback *out_summary) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_summary == nullptr || state->metrics_summary == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  return static_cast<std::int32_t>(moonai_gpu::copy_compact_device_readback(state->metrics_summary, out_summary,
                                                                            sizeof(*out_summary)));
}

extern "C" std::int32_t moonai_gpu_simulation_refresh_reports(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->metrics_summary == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto status = refresh_metrics_summary(*state);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_sensor_snapshot(const void *state_ptr, PopulationKind population_kind,
                                                                   std::uint32_t slot,
                                                                   SensorSnapshotReadback *out_snapshot) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_snapshot == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  if (slot >= population.capacity) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto status = moonai_gpu::launch_single_value_readback(out_snapshot, [&](SensorSnapshotReadback *device_snapshot) {
    sensor_snapshot_kernel<<<1U, 1U>>>(population, population_kind, slot, state->config.num_inputs, device_snapshot);
    return CudaStatus::Success;
  });
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_render_snapshot(const void *state_ptr, std::uint32_t max_predators,
                                                                 std::uint32_t max_prey, std::uint32_t max_food,
                                                                 RenderSnapshotHeader *out_header,
                                                                RenderAgentReadback *out_predators,
                                                                RenderAgentReadback *out_prey,
                                                                RenderFoodReadback *out_food) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_header == nullptr ||
      (max_predators != 0U && out_predators == nullptr) || (max_prey != 0U && out_prey == nullptr) ||
      (max_food != 0U && out_food == nullptr) || state->counters == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  RenderSnapshotHeader *device_header = nullptr;
  RenderAgentReadback *device_predators = nullptr;
  RenderAgentReadback *device_prey = nullptr;
  RenderFoodReadback *device_food = nullptr;
  auto status = moonai_gpu::alloc_array(&device_header, 1U);
  if (status == CudaStatus::Success && max_predators > 0U) {
    status = moonai_gpu::alloc_array(&device_predators, max_predators);
  }
  if (status == CudaStatus::Success && max_prey > 0U) {
    status = moonai_gpu::alloc_array(&device_prey, max_prey);
  }
  if (status == CudaStatus::Success && max_food > 0U) {
    status = moonai_gpu::alloc_array(&device_food, max_food);
  }
  if (status == CudaStatus::Success) {
    render_snapshot_kernel<<<1U, 1U>>>(state->predator, state->prey, state->food, state->counters, max_predators, max_prey,
                                       max_food, device_header, device_predators, device_prey, device_food);
    status = moonai_gpu::synchronize_kernels();
  }
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(device_header, out_header, sizeof(*out_header));
  }
  if (status == CudaStatus::Success && out_header->returned_predators > 0U) {
    status = moonai_gpu::copy_compact_device_readback(device_predators, out_predators,
                                                      sizeof(RenderAgentReadback) * out_header->returned_predators);
  }
  if (status == CudaStatus::Success && out_header->returned_prey > 0U) {
    status = moonai_gpu::copy_compact_device_readback(device_prey, out_prey,
                                                      sizeof(RenderAgentReadback) * out_header->returned_prey);
  }
  if (status == CudaStatus::Success && out_header->returned_food > 0U) {
    status = moonai_gpu::copy_compact_device_readback(device_food, out_food,
                                                      sizeof(RenderFoodReadback) * out_header->returned_food);
  }

  moonai_gpu::free_array(device_header);
  moonai_gpu::free_array(device_predators);
  moonai_gpu::free_array(device_prey);
  moonai_gpu::free_array(device_food);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_last_cuda_error_code() { return moonai_gpu::g_last_cuda_error_code; }
