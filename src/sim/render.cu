#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::FoodBuffer;
using moonai_gpu::PopulationKind;
using moonai_gpu::RenderAgentReadback;
using moonai_gpu::RenderFoodReadback;
using moonai_gpu::RenderSnapshotHeader;
using moonai_gpu::SimulationCounters;

namespace {

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

} // namespace

extern "C" std::int32_t dev_initialize_render_snapshot(const DeviceState *state) {
  initialize_render_snapshot_kernel<<<1U, 1U>>>(state->counters, state->render_header_scratch);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_pack_render_agents(const DeviceState *state, PopulationKind population_kind,
                                                std::uint32_t max_count) {
  const auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  auto *scratch = population_kind == PopulationKind::Predator ? state->render_predators_scratch : state->render_prey_scratch;
  const auto blocks = (population.capacity + 255U) / 256U;
  if (population_kind == PopulationKind::Predator) {
    pack_render_agents_kernel<PopulationKind::Predator><<<blocks == 0U ? 1U : blocks, 256U>>>(
        population, max_count, state->render_header_scratch, scratch);
  } else {
    pack_render_agents_kernel<PopulationKind::Prey><<<blocks == 0U ? 1U : blocks, 256U>>>(population, max_count,
                                                                                            state->render_header_scratch,
                                                                                            scratch);
  }
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_pack_render_food(const DeviceState *state, std::uint32_t max_food) {
  const auto food_blocks = (state->food.capacity + 255U) / 256U;
  pack_render_food_kernel<<<food_blocks == 0U ? 1U : food_blocks, 256U>>>(state->food, max_food,
                                                                            state->render_header_scratch,
                                                                            state->render_food_scratch);
  return moonai_gpu::synchronize_kernels();
}
