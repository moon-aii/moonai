#include "evolution_cuda.cuh"

#include <new>
#include <tuple>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

using moonai_gpu::CudaStatus;
using moonai_gpu::DeviceInnovationState;
using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::FoodBuffer;
using moonai_gpu::GpuEvolutionConfig;
using moonai_gpu::GpuEvolutionState;
using moonai_gpu::PopulationKind;
using moonai_gpu::RenderAgentReadback;
using moonai_gpu::RenderFoodReadback;
using moonai_gpu::RenderSnapshotHeader;
using moonai_gpu::ReproductionPairReadback;
using moonai_gpu::SensorSnapshotReadback;
using moonai_gpu::SimulationCounters;
using moonai_gpu::SimulationConfig;
using moonai_gpu::UiStatsReadback;
using moonai_gpu::PopulationGridEntry;
using moonai_gpu::FoodGridEntry;
using moonai_gpu::MetricsReduceScratch;
using moonai_gpu::MetricsSummaryReadback;

namespace moonai_gpu {

std::int32_t g_last_cuda_error_code = 0;

} // namespace moonai_gpu

extern "C" std::int32_t moonai_gpu_evolution_crossover(void *state_ptr, PopulationKind population_kind,
                                                         std::uint32_t parent_a_slot, std::uint32_t parent_b_slot,
                                                         std::uint32_t offspring_slot);
extern "C" std::int32_t moonai_gpu_evolution_population_live_count(const void *state_ptr, PopulationKind population_kind,
                                                                      std::uint32_t *out_live_count);
extern "C" std::int32_t moonai_gpu_evolution_mutate_slot(void *state_ptr, PopulationKind population_kind,
                                                           std::uint32_t slot,
                                                           const moonai_gpu::GpuMutationConfig *config);
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

void free_interaction_buffers(GpuEvolutionState &state) {
  moonai_gpu::free_array(state.food_claimed_by);
  moonai_gpu::free_array(state.prey_claimed_by);
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
}

CudaStatus ensure_reproduction_buffers(GpuEvolutionState &state) {
  if (state.predator_mate_claims != nullptr && state.prey_mate_claims != nullptr && state.predator_pair_count != nullptr &&
      state.prey_pair_count != nullptr && (state.predator.capacity == 0U || state.predator_reproduction_pairs != nullptr) &&
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
  moonai_gpu::free_array(state->next_entity_id);
  moonai_gpu::free_array(state->counters);
  moonai_gpu::free_array(state->predator_free_list);
  moonai_gpu::free_array(state->prey_free_list);
  moonai_gpu::free_array(state->predator_free_len);
  moonai_gpu::free_array(state->prey_free_len);
  free_reproduction_buffers(*state);
  free_interaction_buffers(*state);
  moonai_gpu::free_array(state->population_live_count_scratch);
  moonai_gpu::free_array(state->ui_stats_scratch);
  moonai_gpu::free_array(state->free_list_state_scratch);
  moonai_gpu::free_array(state->sensor_snapshot_scratch);
  moonai_gpu::free_array(state->compiled_header_scratch);
  moonai_gpu::free_array(state->selected_network_scratch);
  moonai_gpu::free_array(state->metrics_summary);
  moonai_gpu::free_array(state->metrics_reduce_scratch);
  moonai_gpu::free_array(state->species_summaries_scratch);
  moonai_gpu::free_array(state->representative_headers_scratch);
  moonai_gpu::free_array(state->species_count_scratch);
  moonai_gpu::free_array(state->render_header_scratch);
  moonai_gpu::free_array(state->render_predators_scratch);
  moonai_gpu::free_array(state->render_prey_scratch);
  moonai_gpu::free_array(state->render_food_scratch);
  free_spatial_grid_buffers(*state);
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
    moonai_gpu::free_array(state.render_predators_scratch);
  } else {
    moonai_gpu::free_array(state.prey_free_list);
    moonai_gpu::free_array(state.render_prey_scratch);
    moonai_gpu::free_array(state.prey_claimed_by);
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

  if (population_kind == PopulationKind::Predator) {
    status = moonai_gpu::alloc_array(&state.render_predators_scratch, population.capacity);
  } else {
    status = moonai_gpu::alloc_array(&state.render_prey_scratch, population.capacity);
  }
  if (status != CudaStatus::Success) {
    return status;
  }

  if (population_kind == PopulationKind::Prey) {
    status = moonai_gpu::alloc_array(&state.prey_claimed_by, population.capacity);
    if (status != CudaStatus::Success) {
      return status;
    }
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

__global__ void finalize_cell_offsets_kernel(const std::uint32_t *cell_counts, std::uint32_t *cell_offsets,
                                             std::uint32_t *cell_write_offsets, std::uint32_t cell_count) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < cell_count) {
    cell_write_offsets[idx] = cell_offsets[idx];
  }
  if (idx == 0U) {
    cell_offsets[cell_count] = cell_count == 0U ? 0U : cell_offsets[cell_count - 1U] + cell_counts[cell_count - 1U];
  }
}

__global__ void reset_reproduction_state_kernel(std::uint32_t *mate_claims, std::uint32_t capacity,
                                                std::uint32_t *out_pair_count) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < capacity) {
    mate_claims[idx] = moonai_gpu::kUnclaimedMate;
  }
  if (idx == 0U) {
    *out_pair_count = 0U;
  }
}

__global__ void find_reproduction_pairs_kernel(DevicePopulationBuffers population,
                                               const std::uint32_t *cell_offsets,
                                               const PopulationGridEntry *entries,
                                               std::uint32_t grid_cols, std::uint32_t grid_rows,
                                               float grid_cell_size, float mate_range,
                                               float reproduction_energy_threshold, std::uint32_t *mate_claims,
                                               ReproductionPairReadback *out_pairs, std::uint32_t *out_pair_count) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }
  if (population.energy[idx] < reproduction_energy_threshold) {
    return;
  }

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

  if (atomicCAS(&mate_claims[best_mate], moonai_gpu::kUnclaimedMate, idx) != moonai_gpu::kUnclaimedMate) {
    return;
  }

  const auto pair_index = atomicAdd(out_pair_count, 1U);
  out_pairs[pair_index] = ReproductionPairReadback{idx, best_mate};
}

__global__ void apply_reproduction_energy_kernel(DevicePopulationBuffers population,
                                                  const ReproductionPairReadback *pairs,
                                                   std::uint32_t pair_count, float energy_cost,
                                                   std::uint32_t *birth_counter, std::uint32_t *death_counter) {
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
                                    const std::uint32_t *food_cell_offsets, const FoodGridEntry *food_entries,
                                    std::uint32_t *food_claimed_by, std::uint32_t grid_cols, std::uint32_t grid_rows,
                                    float grid_cell_size, float interaction_range) {
  const auto prey_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (prey_idx >= prey.capacity || prey.alive[prey_idx] == 0U) {
    return;
  }

  const auto interaction_range_sq = interaction_range * interaction_range;
  const auto px = prey.pos_x[prey_idx];
  const auto py = prey.pos_y[prey_idx];
  const auto cells_to_check = static_cast<std::int32_t>(interaction_range / grid_cell_size) + 1;
  const auto base_cx = static_cast<std::int32_t>(cell_coord(px, grid_cell_size, grid_cols));
  const auto base_cy = static_cast<std::int32_t>(cell_coord(py, grid_cell_size, grid_rows));

  std::uint32_t best_food = food.capacity;
  float best_distance_sq = interaction_range_sq;
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
                                     px, py, interaction_range)) {
        continue;
      }

      const auto cell = (static_cast<std::uint32_t>(cy) * grid_cols) + static_cast<std::uint32_t>(cx);
      for (auto slot = food_cell_offsets[cell]; slot < food_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = food_entries[slot];
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto distance_sq = (dx * dx) + (dy * dy);
        if (distance_sq > best_distance_sq) {
          continue;
        }
        best_distance_sq = distance_sq;
        best_food = entry.slot;
      }
    }
  }

  if (best_food < food.capacity) {
    atomicMin(&food_claimed_by[best_food], prey_idx);
  }
}

__global__ void finalize_food_kernel(DevicePopulationBuffers prey, FoodBuffer food, SimulationCounters *counters,
                                     const std::uint32_t *food_claimed_by, float energy_gain_from_food,
                                     float max_energy) {
  const auto food_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (food_idx >= food.capacity || food.active[food_idx] == 0U) {
    return;
  }

  const auto prey_idx = food_claimed_by[food_idx];
  if (prey_idx == moonai_gpu::kUnclaimedMate || prey_idx >= prey.capacity || prey.alive[prey_idx] == 0U) {
    return;
  }

  food.active[food_idx] = 0U;
  prey.energy[prey_idx] += energy_gain_from_food;
  if (prey.energy[prey_idx] > max_energy) {
    prey.energy[prey_idx] = max_energy;
  }
  atomicAdd(&counters->food_eaten, 1U);
}

__global__ void respawn_food_kernel(FoodBuffer food, const SimulationCounters *counters, std::uint64_t base_seed,
                                    float respawn_rate, float world_size) {
  const auto food_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (food_idx >= food.capacity || food.active[food_idx] != 0U) {
    return;
  }

  auto rng = moonai_gpu::splitmix64(base_seed ^ (static_cast<std::uint64_t>(counters->tick) << 32U) ^ food_idx);
  if (moonai_gpu::next_unit_float(rng) >= respawn_rate) {
    return;
  }

  food.pos_x[food_idx] = moonai_gpu::next_unit_float(rng) * world_size;
  food.pos_y[food_idx] = moonai_gpu::next_unit_float(rng) * world_size;
  food.active[food_idx] = 1U;
}

__global__ void resolve_combat_kernel(DevicePopulationBuffers predator, DevicePopulationBuffers prey,
                                      std::uint32_t *prey_claimed_by, const std::uint32_t *prey_cell_offsets,
                                      const PopulationGridEntry *prey_entries, std::uint32_t grid_cols,
                                      std::uint32_t grid_rows, float grid_cell_size, float interaction_range) {
  const auto predator_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (predator_idx >= predator.capacity || predator.alive[predator_idx] == 0U) {
    return;
  }

  const auto interaction_range_sq = interaction_range * interaction_range;
  const auto px = predator.pos_x[predator_idx];
  const auto py = predator.pos_y[predator_idx];
  const auto cells_to_check = static_cast<std::int32_t>(interaction_range / grid_cell_size) + 1;
  const auto base_cx = static_cast<std::int32_t>(cell_coord(px, grid_cell_size, grid_cols));
  const auto base_cy = static_cast<std::int32_t>(cell_coord(py, grid_cell_size, grid_rows));

  std::uint32_t best_prey = prey.capacity;
  float best_distance_sq = interaction_range_sq;
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
                                     px, py, interaction_range)) {
        continue;
      }

      const auto cell = (static_cast<std::uint32_t>(cy) * grid_cols) + static_cast<std::uint32_t>(cx);
      for (auto slot = prey_cell_offsets[cell]; slot < prey_cell_offsets[cell + 1U]; ++slot) {
        const auto entry = prey_entries[slot];
        const auto dx = entry.pos_x - px;
        const auto dy = entry.pos_y - py;
        const auto distance_sq = (dx * dx) + (dy * dy);
        if (distance_sq > best_distance_sq) {
          continue;
        }
        best_distance_sq = distance_sq;
        best_prey = entry.slot;
      }
    }
  }

  if (best_prey < prey.capacity) {
    atomicMin(&prey_claimed_by[best_prey], predator_idx);
  }
}

__global__ void finalize_combat_kernel(DevicePopulationBuffers predator, DevicePopulationBuffers prey,
                                       SimulationCounters *counters, std::uint32_t *prey_free_list,
                                       std::uint32_t *prey_free_len, const std::uint32_t *prey_claimed_by,
                                       float energy_gain_from_kill, float max_energy) {
  const auto prey_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (prey_idx >= prey.capacity || prey.alive[prey_idx] == 0U) {
    return;
  }

  const auto predator_idx = prey_claimed_by[prey_idx];
  if (predator_idx == moonai_gpu::kUnclaimedMate || predator_idx >= predator.capacity ||
      predator.alive[predator_idx] == 0U) {
    return;
  }

  prey.alive[prey_idx] = 0U;
  prey.energy[prey_idx] = 0.0F;
  prey.vel_x[prey_idx] = 0.0F;
  prey.vel_y[prey_idx] = 0.0F;
  const auto free_index = atomicAdd(prey_free_len, 1U);
  if (free_index < prey.capacity) {
    prey_free_list[free_index] = prey_idx;
  }
  predator.energy[predator_idx] += energy_gain_from_kill;
  if (predator.energy[predator_idx] > max_energy) {
    predator.energy[predator_idx] = max_energy;
  }
  atomicAdd(&counters->kills, 1U);
  atomicAdd(&counters->prey_deaths, 1U);
}

__global__ void apply_movement_kernel(DevicePopulationBuffers population, float speed, float world_size) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  population.pos_x[idx] = clamp_world(population.pos_x[idx] + (population.vel_x[idx] * speed), world_size);
  population.pos_y[idx] = clamp_world(population.pos_y[idx] + (population.vel_y[idx] * speed), world_size);
}

template <PopulationKind Kind>
__global__ void accumulate_metrics_kernel(DevicePopulationBuffers population, MetricsReduceScratch *scratch) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  const auto complexity = static_cast<float>(population.genome.num_nodes[idx] +
                                             moonai_gpu::count_enabled_connections(population, idx,
                                                                                   population.genome.num_connections[idx]));
  const auto generation = population.generation[idx];
  const auto species_id = population.species_id[idx];
  if constexpr (Kind == PopulationKind::Predator) {
    atomicAdd(&scratch->predator_count, 1U);
    atomicAdd(&scratch->predator_energy_sum, population.energy[idx]);
    atomicAdd(&scratch->predator_complexity_sum, complexity);
    atomicAdd(&scratch->predator_generation_sum, static_cast<float>(generation));
    atomicMax(&scratch->max_predator_generation, generation);
    if (species_id < moonai_gpu::kSpeciesBucketCount) {
      atomicOr(&scratch->predator_species_mask, 1ULL << species_id);
    }
  } else {
    atomicAdd(&scratch->prey_count, 1U);
    atomicAdd(&scratch->prey_energy_sum, population.energy[idx]);
    atomicAdd(&scratch->prey_complexity_sum, complexity);
    atomicAdd(&scratch->prey_generation_sum, static_cast<float>(generation));
    atomicMax(&scratch->max_prey_generation, generation);
    if (species_id < moonai_gpu::kSpeciesBucketCount) {
      atomicOr(&scratch->prey_species_mask, 1ULL << species_id);
    }
  }
}

__global__ void finalize_metrics_reduce_kernel(const SimulationCounters *counters, const MetricsReduceScratch *scratch,
                                               MetricsSummaryReadback *out_metrics) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  const auto predator_count = scratch->predator_count;
  const auto prey_count = scratch->prey_count;
  out_metrics->tick = counters->tick;
  out_metrics->predator_count = predator_count;
  out_metrics->prey_count = prey_count;
  out_metrics->predator_births = counters->predator_births;
  out_metrics->prey_births = counters->prey_births;
  out_metrics->predator_deaths = counters->predator_deaths;
  out_metrics->prey_deaths = counters->prey_deaths;
  out_metrics->predator_species = static_cast<std::uint32_t>(__popcll(scratch->predator_species_mask));
  out_metrics->prey_species = static_cast<std::uint32_t>(__popcll(scratch->prey_species_mask));
  out_metrics->avg_predator_complexity =
      predator_count == 0U ? 0.0F : scratch->predator_complexity_sum / static_cast<float>(predator_count);
  out_metrics->avg_prey_complexity = prey_count == 0U ? 0.0F : scratch->prey_complexity_sum / static_cast<float>(prey_count);
  out_metrics->avg_predator_energy = predator_count == 0U ? 0.0F : scratch->predator_energy_sum / static_cast<float>(predator_count);
  out_metrics->avg_prey_energy = prey_count == 0U ? 0.0F : scratch->prey_energy_sum / static_cast<float>(prey_count);
  out_metrics->max_predator_generation = scratch->max_predator_generation;
  out_metrics->avg_predator_generation =
      predator_count == 0U ? 0.0F : scratch->predator_generation_sum / static_cast<float>(predator_count);
  out_metrics->max_prey_generation = scratch->max_prey_generation;
  out_metrics->avg_prey_generation = prey_count == 0U ? 0.0F : scratch->prey_generation_sum / static_cast<float>(prey_count);
}

__global__ void write_ui_stats_kernel(DevicePopulationBuffers predator, DevicePopulationBuffers prey,
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

__global__ void initialize_render_snapshot_kernel(const SimulationCounters *counters, RenderSnapshotHeader *out_header) {
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
}

template <PopulationKind Kind>
__global__ void pack_render_agents_kernel(DevicePopulationBuffers population, std::uint32_t max_count,
                                          RenderSnapshotHeader *out_header, RenderAgentReadback *out_agents) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= population.capacity || population.alive[idx] == 0U) {
    return;
  }

  auto *total_ptr = Kind == PopulationKind::Predator ? &out_header->total_predators : &out_header->total_prey;
  auto *returned_ptr = Kind == PopulationKind::Predator ? &out_header->returned_predators : &out_header->returned_prey;
  atomicAdd(total_ptr, 1U);
  const auto write_index = atomicAdd(returned_ptr, 1U);
  if (write_index < max_count) {
    out_agents[write_index] = RenderAgentReadback{Kind,
                                                  idx,
                                                  population.entity_id[idx],
                                                  population.species_id[idx],
                                                  population.generation[idx],
                                                  population.age[idx],
                                                  population.pos_x[idx],
                                                  population.pos_y[idx],
                                                  population.vel_x[idx],
                                                  population.vel_y[idx],
                                                  population.energy[idx]};
  }
}

__global__ void pack_render_food_kernel(FoodBuffer food, std::uint32_t max_food, RenderSnapshotHeader *out_header,
                                        RenderFoodReadback *out_food) {
  const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= food.capacity || food.active[idx] == 0U) {
    return;
  }

  atomicAdd(&out_header->total_food, 1U);
  const auto write_index = atomicAdd(&out_header->returned_food, 1U);
  if (write_index < max_food) {
    out_food[write_index] = RenderFoodReadback{idx, food.active[idx], 0U, 0U, food.pos_x[idx], food.pos_y[idx]};
  }
}

__global__ void summarize_population_kernel(DevicePopulationBuffers population, std::uint32_t *out_live_count) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  std::uint32_t live_count = 0U;
  for (std::uint32_t idx = 0; idx < population.capacity; ++idx) {
    if (population.alive[idx] == 0U) {
      continue;
    }
    ++live_count;
  }

  *out_live_count = live_count;
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
  thrust::exclusive_scan(thrust::device, state.predator_cell_counts, state.predator_cell_counts + cell_count,
                         state.predator_cell_offsets);
  thrust::exclusive_scan(thrust::device, state.prey_cell_counts, state.prey_cell_counts + cell_count,
                         state.prey_cell_offsets);
  thrust::exclusive_scan(thrust::device, state.food_cell_counts, state.food_cell_counts + cell_count,
                         state.food_cell_offsets);
  const auto cell_blocks = (cell_count + 255U) / 256U;
  finalize_cell_offsets_kernel<<<cell_blocks == 0U ? 1U : cell_blocks, 256U>>>(
      state.predator_cell_counts, state.predator_cell_offsets, state.predator_cell_write_offsets, cell_count);
  finalize_cell_offsets_kernel<<<cell_blocks == 0U ? 1U : cell_blocks, 256U>>>(
      state.prey_cell_counts, state.prey_cell_offsets, state.prey_cell_write_offsets, cell_count);
  finalize_cell_offsets_kernel<<<cell_blocks == 0U ? 1U : cell_blocks, 256U>>>(
      state.food_cell_counts, state.food_cell_offsets, state.food_cell_write_offsets, cell_count);
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
  status = moonai_gpu::zero_device_memory(state.metrics_reduce_scratch, sizeof(MetricsReduceScratch));
  if (status != CudaStatus::Success) {
    return status;
  }
  const auto predator_blocks = (state.predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state.prey.capacity + 255U) / 256U;
  accumulate_metrics_kernel<PopulationKind::Predator><<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state.predator, state.metrics_reduce_scratch);
  accumulate_metrics_kernel<PopulationKind::Prey><<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state.prey, state.metrics_reduce_scratch);
  finalize_metrics_reduce_kernel<<<1U, 1U>>>(state.counters, state.metrics_reduce_scratch, state.metrics_summary);
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

  for (auto status : {
           allocate_population_buffers(state->predator, config->predator_capacity, config->num_inputs, config->node_stride,
                                       config->connection_stride, config->num_outputs),
           allocate_population_buffers(state->prey, config->prey_capacity, config->num_inputs, config->node_stride,
                                       config->connection_stride, config->num_outputs),
           moonai_gpu::alloc_array(&state->innovation, 1U),
           moonai_gpu::alloc_array(&state->next_entity_id, 1U),
            moonai_gpu::alloc_array(&state->population_live_count_scratch, 1U),
            moonai_gpu::alloc_array(&state->ui_stats_scratch, 1U),
            moonai_gpu::alloc_array(&state->free_list_state_scratch, 1U),
            moonai_gpu::alloc_array(&state->sensor_snapshot_scratch, 1U),
            moonai_gpu::alloc_array(&state->compiled_header_scratch, 1U),
            moonai_gpu::alloc_array(&state->selected_network_scratch, 1U),
            moonai_gpu::alloc_array(&state->metrics_summary, 1U),
            moonai_gpu::alloc_array(&state->metrics_reduce_scratch, 1U),
            moonai_gpu::alloc_array(&state->species_summaries_scratch, moonai_gpu::kSpeciesBucketCount),
            moonai_gpu::alloc_array(&state->representative_headers_scratch, moonai_gpu::kSpeciesBucketCount),
           moonai_gpu::alloc_array(&state->species_count_scratch, 1U),
           moonai_gpu::alloc_array(&state->render_header_scratch, 1U),
           moonai_gpu::alloc_array(&state->render_predators_scratch, config->predator_capacity),
           moonai_gpu::alloc_array(&state->render_prey_scratch, config->prey_capacity),
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

  const DeviceInnovationState innovation_state{
      (state->config.num_inputs + 1U) * state->config.num_outputs,
      state->config.num_inputs + state->config.num_outputs + 1U,
  };
  status = moonai_gpu::copy_host_data_to_device(state->innovation, &innovation_state, sizeof(innovation_state));
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

extern "C" std::int32_t moonai_gpu_evolution_population_live_count(const void *state_ptr, PopulationKind population_kind,
                                                                     std::uint32_t *out_live_count) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_live_count == nullptr || state->population_live_count_scratch == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  summarize_population_kernel<<<1U, 1U>>>(moonai_gpu::population_for_kind(*state, population_kind),
                                          state->population_live_count_scratch);
  auto status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(state->population_live_count_scratch, out_live_count,
                                                      sizeof(*out_live_count));
  }
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_set_config(void *state_ptr, const SimulationConfig *config) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || config == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  state->simulation = *config;
  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" std::int32_t moonai_gpu_simulation_set_grid(void *state_ptr, float cell_size, std::uint32_t grid_cols,
                                                         std::uint32_t grid_rows) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || cell_size <= 0.0F || grid_cols == 0U || grid_rows == 0U) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  state->grid_cell_size = cell_size;
  state->grid_cols = grid_cols;
  state->grid_rows = grid_rows;
  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" std::int32_t moonai_gpu_simulation_ensure_food_buffer(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  if (state->food.capacity != state->simulation.food_count) {
    free_food_buffers(state->food);
    moonai_gpu::free_array(state->render_food_scratch);
    moonai_gpu::free_array(state->food_claimed_by);
  }
  if (state->simulation.food_count > 0U && state->food.capacity == 0U) {
    auto status = allocate_food_buffers(state->food, state->simulation.food_count);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
    status = moonai_gpu::alloc_array(&state->render_food_scratch, state->simulation.food_count);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  if (state->food.capacity == 0U) {
    moonai_gpu::free_array(state->food_claimed_by);
  } else if (state->food_claimed_by == nullptr) {
    const auto status = moonai_gpu::alloc_array(&state->food_claimed_by, state->food.capacity);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" std::int32_t moonai_gpu_simulation_ensure_counter_buffer(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  if (state->counters == nullptr) {
    return static_cast<std::int32_t>(moonai_gpu::alloc_array(&state->counters, 1U));
  }
  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" std::int32_t moonai_gpu_simulation_ensure_free_lists(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
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
  if (state->prey.capacity == 0U) {
    moonai_gpu::free_array(state->prey_claimed_by);
  } else if (state->prey_claimed_by == nullptr) {
    const auto status = moonai_gpu::alloc_array(&state->prey_claimed_by, state->prey.capacity);
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }
  return static_cast<std::int32_t>(CudaStatus::Success);
}

extern "C" std::int32_t moonai_gpu_simulation_ensure_reproduction_buffers(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  return static_cast<std::int32_t>(ensure_reproduction_buffers(*state));
}

extern "C" std::int32_t moonai_gpu_simulation_ensure_metrics_buffer(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  return static_cast<std::int32_t>(ensure_metrics_buffer(*state));
}

extern "C" std::int32_t moonai_gpu_simulation_ensure_spatial_grid_buffers(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->grid_cols == 0U || state->grid_rows == 0U) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  return static_cast<std::int32_t>(ensure_spatial_grid_buffers(*state, state->grid_cols * state->grid_rows));
}

extern "C" std::int32_t moonai_gpu_simulation_reset_counters(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->counters == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  reset_simulation_counters_kernel<<<1U, 1U>>>(state->counters);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_initialize_free_lists(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->predator_free_list == nullptr || state->prey_free_list == nullptr ||
      state->predator_free_len == nullptr || state->prey_free_len == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  initialize_free_list_kernel<<<1U, 1U>>>(state->predator, state->predator_free_list, state->predator_free_len);
  initialize_free_list_kernel<<<1U, 1U>>>(state->prey, state->prey_free_list, state->prey_free_len);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_seed_food(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  if (state->food.capacity == 0U) {
    return static_cast<std::int32_t>(CudaStatus::Success);
  }
  const auto food_blocks = (state->food.capacity + 255U) / 256U;
  seed_food_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(state->food, state->simulation.seed ^ 0xC0FFEEULL,
                                                                    state->simulation.grid_size);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_reset_reproduction_state(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->predator_mate_claims == nullptr || state->prey_mate_claims == nullptr ||
      state->predator_pair_count == nullptr || state->prey_pair_count == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  reset_reproduction_state_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator_mate_claims, state->predator.capacity, state->predator_pair_count);
  reset_reproduction_state_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey_mate_claims, state->prey.capacity, state->prey_pair_count);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_build_spatial_grid(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  return static_cast<std::int32_t>(build_spatial_grid(*state));
}

extern "C" std::int32_t moonai_gpu_simulation_compute_sensor_inputs(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  compute_sensor_inputs_kernel<true><<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->config.num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.predator_speed, state->simulation.grid_size);
  compute_sensor_inputs_kernel<false><<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, state->predator_cell_offsets, state->predator_grid_entries, state->prey_cell_offsets,
      state->prey_grid_entries, state->food_cell_offsets, state->food_grid_entries, state->grid_cols, state->grid_rows,
      state->grid_cell_size, state->config.num_inputs, state->simulation.vision_range, state->simulation.max_energy,
      state->simulation.prey_speed, state->simulation.grid_size);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_infer_population(void *state_ptr, PopulationKind population_kind) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  const auto blocks = (population.capacity + 255U) / 256U;
  infer_population_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, state->config.num_inputs);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_update_vitals(void *state_ptr, PopulationKind population_kind) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *free_list = population_kind == PopulationKind::Predator ? state->predator_free_list : state->prey_free_list;
  auto *free_len = population_kind == PopulationKind::Predator ? state->predator_free_len : state->prey_free_len;
  auto *death_counter = population_kind == PopulationKind::Predator ? &state->counters->predator_deaths : &state->counters->prey_deaths;
  const auto blocks = (population.capacity + 255U) / 256U;
  update_vitals_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, state->simulation.energy_drain_per_tick,
                                                              state->simulation.max_age, free_list, free_len, death_counter);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_resolve_food(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->counters == nullptr || state->food_claimed_by == nullptr ||
      state->food_cell_offsets == nullptr || state->food_grid_entries == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  if (state->prey.capacity == 0U || state->food.capacity == 0U) {
    return static_cast<std::int32_t>(CudaStatus::Success);
  }
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  const auto food_blocks = (state->food.capacity + 255U) / 256U;
  const auto memset_status = moonai_gpu::map_cuda_runtime_error(
      cudaMemset(state->food_claimed_by, 0xFF, sizeof(std::uint32_t) * state->food.capacity), CudaStatus::DeviceCopyFailed);
  if (memset_status != CudaStatus::Success) {
    return static_cast<std::int32_t>(memset_status);
  }
  resolve_food_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, state->food, state->counters, state->food_cell_offsets, state->food_grid_entries,
      state->food_claimed_by, state->grid_cols, state->grid_rows, state->grid_cell_size, state->simulation.interaction_range);
  finalize_food_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(
      state->prey, state->food, state->counters, state->food_claimed_by, state->simulation.energy_gain_from_food,
      state->simulation.max_energy);
  respawn_food_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(state->food, state->counters,
                                                                       state->simulation.seed ^ 0xC0FFEEULL,
                                                                       state->simulation.food_respawn_rate,
                                                                       state->simulation.grid_size);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_resolve_combat(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->counters == nullptr || state->prey_free_list == nullptr ||
      state->prey_free_len == nullptr || state->prey_claimed_by == nullptr || state->prey_cell_offsets == nullptr ||
      state->prey_grid_entries == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  if (state->predator.capacity == 0U || state->prey.capacity == 0U) {
    return static_cast<std::int32_t>(CudaStatus::Success);
  }
  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  const auto memset_status = moonai_gpu::map_cuda_runtime_error(
      cudaMemset(state->prey_claimed_by, 0xFF, sizeof(std::uint32_t) * state->prey.capacity), CudaStatus::DeviceCopyFailed);
  if (memset_status != CudaStatus::Success) {
    return static_cast<std::int32_t>(memset_status);
  }
  resolve_combat_kernel<<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, state->prey, state->prey_claimed_by, state->prey_cell_offsets, state->prey_grid_entries,
      state->grid_cols, state->grid_rows, state->grid_cell_size, state->simulation.interaction_range);
  finalize_combat_kernel<<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->predator, state->prey, state->counters, state->prey_free_list, state->prey_free_len, state->prey_claimed_by,
      state->simulation.energy_gain_from_kill, state->simulation.max_energy);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_apply_movement(void *state_ptr, PopulationKind population_kind) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  const auto speed = population_kind == PopulationKind::Predator ? state->simulation.predator_speed : state->simulation.prey_speed;
  const auto blocks = (population.capacity + 255U) / 256U;
  apply_movement_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(population, speed, state->simulation.grid_size);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_reproduction_candidate_count(void *state_ptr,
                                                                             PopulationKind population_kind,
                                                                             std::uint32_t *out_pair_count) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_pair_count == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *mate_claims = population_kind == PopulationKind::Predator ? state->predator_mate_claims : state->prey_mate_claims;
  auto *pair_buffer = population_kind == PopulationKind::Predator ? state->predator_reproduction_pairs : state->prey_reproduction_pairs;
  auto *pair_count_ptr = population_kind == PopulationKind::Predator ? state->predator_pair_count : state->prey_pair_count;
  auto *cell_offsets = population_kind == PopulationKind::Predator ? state->predator_cell_offsets : state->prey_cell_offsets;
  auto *entries = population_kind == PopulationKind::Predator ? state->predator_grid_entries : state->prey_grid_entries;
  const auto blocks = (population.capacity + 255U) / 256U;
  reset_reproduction_state_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(mate_claims, population.capacity, pair_count_ptr);
  find_reproduction_pairs_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(
      population, cell_offsets, entries, state->grid_cols, state->grid_rows, state->grid_cell_size, state->simulation.mate_range,
      state->simulation.reproduction_energy_threshold, mate_claims, pair_buffer, pair_count_ptr);
  auto status = moonai_gpu::synchronize_kernels();
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }
  status = read_device_u32(pair_count_ptr, *out_pair_count);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_read_reproduction_pairs(const void *state_ptr,
                                                                        PopulationKind population_kind,
                                                                        std::uint32_t max_pairs,
                                                                        ReproductionPairReadback *out_pairs,
                                                                        std::uint32_t *out_returned_pairs) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_returned_pairs == nullptr || (max_pairs != 0U && out_pairs == nullptr)) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto *pair_count_ptr = population_kind == PopulationKind::Predator ? state->predator_pair_count : state->prey_pair_count;
  const auto *pair_buffer =
      population_kind == PopulationKind::Predator ? state->predator_reproduction_pairs : state->prey_reproduction_pairs;
  std::uint32_t pair_count = 0U;
  auto status = read_device_u32(pair_count_ptr, pair_count);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  const auto returned_pairs = pair_count < max_pairs ? pair_count : max_pairs;
  *out_returned_pairs = returned_pairs;
  if (returned_pairs == 0U) {
    return static_cast<std::int32_t>(CudaStatus::Success);
  }

  status = moonai_gpu::copy_compact_device_readback(pair_buffer, out_pairs,
                                                    sizeof(ReproductionPairReadback) * returned_pairs);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_read_free_slots(const void *state_ptr, PopulationKind population_kind,
                                                               std::uint32_t max_slots, std::uint32_t *out_slots,
                                                               std::uint32_t *out_returned_slots) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_returned_slots == nullptr || (max_slots != 0U && out_slots == nullptr)) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto *free_len_ptr = population_kind == PopulationKind::Predator ? state->predator_free_len : state->prey_free_len;
  const auto *free_list = population_kind == PopulationKind::Predator ? state->predator_free_list : state->prey_free_list;
  std::uint32_t free_len = 0U;
  auto status = read_device_u32(free_len_ptr, free_len);
  if (status != CudaStatus::Success) {
    return static_cast<std::int32_t>(status);
  }

  const auto returned_slots = free_len < max_slots ? free_len : max_slots;
  *out_returned_slots = returned_slots;
  if (returned_slots == 0U) {
    return static_cast<std::int32_t>(CudaStatus::Success);
  }

  status = moonai_gpu::copy_compact_device_readback(free_list, out_slots, sizeof(std::uint32_t) * returned_slots);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_apply_reproduction_energy(void *state_ptr,
                                                                          PopulationKind population_kind,
                                                                          std::uint32_t births_applied) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->counters == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *pair_buffer = population_kind == PopulationKind::Predator ? state->predator_reproduction_pairs : state->prey_reproduction_pairs;
  auto *birth_counter = population_kind == PopulationKind::Predator ? &state->counters->predator_births : &state->counters->prey_births;
  auto *death_counter = population_kind == PopulationKind::Predator ? &state->counters->predator_deaths : &state->counters->prey_deaths;
  if (births_applied > 0U) {
    const auto blocks = (births_applied + 255U) / 256U;
    apply_reproduction_energy_kernel<<<blocks == 0U ? 1U : blocks, 256U>>>(
        population, pair_buffer, births_applied, state->simulation.reproduction_energy_cost, birth_counter, death_counter);
    auto status = moonai_gpu::synchronize_kernels();
    if (status != CudaStatus::Success) {
      return static_cast<std::int32_t>(status);
    }
  }

  auto *free_list = population_kind == PopulationKind::Predator ? state->predator_free_list : state->prey_free_list;
  auto *free_len = population_kind == PopulationKind::Predator ? state->predator_free_len : state->prey_free_len;
  initialize_free_list_kernel<<<1U, 1U>>>(population, free_list, free_len);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_expand_population(void *state_ptr, PopulationKind population_kind,
                                                                  std::uint32_t new_capacity) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  return static_cast<std::int32_t>(expand_population_capacity(*state, population_kind, new_capacity));
}

extern "C" std::int32_t moonai_gpu_simulation_advance_tick(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->counters == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  advance_tick_kernel<<<1U, 1U>>>(state->counters);
  return static_cast<std::int32_t>(moonai_gpu::synchronize_kernels());
}

extern "C" std::int32_t moonai_gpu_simulation_ui_stats(const void *state_ptr, UiStatsReadback *out_stats) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_stats == nullptr || state->counters == nullptr || state->ui_stats_scratch == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  write_ui_stats_kernel<<<1U, 1U>>>(state->predator, state->prey, state->counters, state->ui_stats_scratch);
  auto status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(state->ui_stats_scratch, out_stats, sizeof(*out_stats));
  }
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_free_list_state(const void *state_ptr,
                                                                 moonai_gpu::FreeListStateReadback *out_state) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_state == nullptr || state->counters == nullptr || state->predator_free_len == nullptr ||
      state->prey_free_len == nullptr || state->free_list_state_scratch == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  free_list_state_kernel<<<1U, 1U>>>(state->food, state->counters, state->predator_free_len, state->prey_free_len,
                                     state->free_list_state_scratch);
  auto status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(state->free_list_state_scratch, out_state, sizeof(*out_state));
  }
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_metrics_summary(const void *state_ptr,
                                                                   MetricsSummaryReadback *out_summary) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_summary == nullptr || state->metrics_summary == nullptr ||
      state->metrics_reduce_scratch == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }
  return static_cast<std::int32_t>(moonai_gpu::copy_compact_device_readback(state->metrics_summary, out_summary,
                                                                            sizeof(*out_summary)));
}

extern "C" std::int32_t moonai_gpu_simulation_refresh_reports(void *state_ptr) {
  auto *state = static_cast<GpuEvolutionState *>(state_ptr);
  if (state == nullptr || state->metrics_summary == nullptr || state->metrics_reduce_scratch == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto status = refresh_metrics_summary(*state);
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_simulation_sensor_snapshot(const void *state_ptr, PopulationKind population_kind,
                                                                   std::uint32_t slot,
                                                                   SensorSnapshotReadback *out_snapshot) {
  auto *state = static_cast<const GpuEvolutionState *>(state_ptr);
  if (state == nullptr || out_snapshot == nullptr || state->sensor_snapshot_scratch == nullptr) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  if (slot >= population.capacity) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  sensor_snapshot_kernel<<<1U, 1U>>>(population, population_kind, slot, state->config.num_inputs,
                                     state->sensor_snapshot_scratch);
  auto status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(state->sensor_snapshot_scratch, out_snapshot, sizeof(*out_snapshot));
  }
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
      (max_food != 0U && out_food == nullptr) || state->counters == nullptr || state->render_header_scratch == nullptr ||
      (max_predators != 0U && state->render_predators_scratch == nullptr) ||
      (max_prey != 0U && state->render_prey_scratch == nullptr) || (max_food != 0U && state->render_food_scratch == nullptr)) {
    return static_cast<std::int32_t>(CudaStatus::InvalidArgument);
  }

  const auto predator_blocks = (state->predator.capacity + 255U) / 256U;
  const auto prey_blocks = (state->prey.capacity + 255U) / 256U;
  const auto food_blocks = (state->food.capacity + 255U) / 256U;
  initialize_render_snapshot_kernel<<<1U, 1U>>>(state->counters, state->render_header_scratch);
  pack_render_agents_kernel<PopulationKind::Predator><<<predator_blocks == 0U ? 1U : predator_blocks, 256U>>>(
      state->predator, max_predators, state->render_header_scratch, state->render_predators_scratch);
  pack_render_agents_kernel<PopulationKind::Prey><<<prey_blocks == 0U ? 1U : prey_blocks, 256U>>>(
      state->prey, max_prey, state->render_header_scratch, state->render_prey_scratch);
  pack_render_food_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(state->food, max_food,
                                                                            state->render_header_scratch,
                                                                            state->render_food_scratch);
  auto status = moonai_gpu::synchronize_kernels();
  if (status == CudaStatus::Success) {
    status = moonai_gpu::copy_compact_device_readback(state->render_header_scratch, out_header, sizeof(*out_header));
  }
  if (status == CudaStatus::Success) {
    if (out_header->returned_predators > max_predators) {
      out_header->returned_predators = max_predators;
    }
    if (out_header->returned_prey > max_prey) {
      out_header->returned_prey = max_prey;
    }
    if (out_header->returned_food > max_food) {
      out_header->returned_food = max_food;
    }
  }
  if (status == CudaStatus::Success && out_header->returned_predators > 0U) {
    status = moonai_gpu::copy_compact_device_readback(state->render_predators_scratch, out_predators,
                                                      sizeof(RenderAgentReadback) * out_header->returned_predators);
  }
  if (status == CudaStatus::Success && out_header->returned_prey > 0U) {
    status = moonai_gpu::copy_compact_device_readback(state->render_prey_scratch, out_prey,
                                                      sizeof(RenderAgentReadback) * out_header->returned_prey);
  }
  if (status == CudaStatus::Success && out_header->returned_food > 0U) {
    status = moonai_gpu::copy_compact_device_readback(state->render_food_scratch, out_food,
                                                      sizeof(RenderFoodReadback) * out_header->returned_food);
  }
  return static_cast<std::int32_t>(status);
}

extern "C" std::int32_t moonai_gpu_last_cuda_error_code() { return moonai_gpu::g_last_cuda_error_code; }
