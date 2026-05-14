#include "sim.cuh"

using moonai_gpu::DevicePopulationBuffers;
using moonai_gpu::DeviceState;
using moonai_gpu::FoodBuffer;
using moonai_gpu::MetricsReduceScratch;
using moonai_gpu::MetricsSummaryReadback;
using moonai_gpu::PopulationKind;
using moonai_gpu::SimulationCounters;
using moonai_gpu::UiStatsReadback;

namespace {

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
      atomicOr((unsigned long long *)&scratch->predator_species_mask, 1ULL << species_id);
    }
  } else {
    atomicAdd(&scratch->prey_count, 1U);
    atomicAdd(&scratch->prey_energy_sum, population.energy[idx]);
    atomicAdd(&scratch->prey_complexity_sum, complexity);
    atomicAdd(&scratch->prey_generation_sum, static_cast<float>(generation));
    atomicMax(&scratch->max_prey_generation, generation);
    if (species_id < moonai_gpu::kSpeciesBucketCount) {
      atomicOr((unsigned long long *)&scratch->prey_species_mask, 1ULL << species_id);
    }
  }
}

__global__ void finalize_metrics_reduce_kernel(const SimulationCounters *counters,
                                               const MetricsReduceScratch *scratch,
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
  out_metrics->avg_prey_complexity =
      prey_count == 0U ? 0.0F : scratch->prey_complexity_sum / static_cast<float>(prey_count);
  out_metrics->avg_predator_energy =
      predator_count == 0U ? 0.0F : scratch->predator_energy_sum / static_cast<float>(predator_count);
  out_metrics->avg_prey_energy = prey_count == 0U ? 0.0F : scratch->prey_energy_sum / static_cast<float>(prey_count);
  out_metrics->max_predator_generation = scratch->max_predator_generation;
  out_metrics->avg_predator_generation =
      predator_count == 0U ? 0.0F : scratch->predator_generation_sum / static_cast<float>(predator_count);
  out_metrics->max_prey_generation = scratch->max_prey_generation;
  out_metrics->avg_prey_generation =
      prey_count == 0U ? 0.0F : scratch->prey_generation_sum / static_cast<float>(prey_count);
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
  out_stats->avg_predator_energy =
      predator_count == 0U ? 0.0F : predator_energy / static_cast<float>(predator_count);
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

} // namespace

extern "C" std::int32_t dev_write_population_live_count(const DeviceState *state, PopulationKind population_kind) {
  summarize_population_kernel<<<1U, 1U>>>(moonai_gpu::population_for_kind(*state, population_kind),
                                          state->population_live_count_scratch);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_write_ui_stats(const DeviceState *state) {
  write_ui_stats_kernel<<<1U, 1U>>>(state->predator, state->prey, state->counters, state->ui_stats_scratch);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_write_free_list_state(const DeviceState *state) {
  free_list_state_kernel<<<1U, 1U>>>(state->food, state->counters, state->predator_free_len, state->prey_free_len,
                                     state->free_list_state_scratch);
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_accumulate_metrics(DeviceState *state, PopulationKind population_kind) {
  auto &population = moonai_gpu::population_for_kind(*state, population_kind);
  const auto blocks = (population.capacity + 255U) / 256U;
  if (population_kind == PopulationKind::Predator) {
    accumulate_metrics_kernel<PopulationKind::Predator><<<blocks == 0U ? 1U : blocks, 256U>>>(population,
                                                                                                 state->metrics_reduce_scratch);
  } else {
    accumulate_metrics_kernel<PopulationKind::Prey><<<blocks == 0U ? 1U : blocks, 256U>>>(population,
                                                                                             state->metrics_reduce_scratch);
  }
  return moonai_gpu::synchronize_kernels();
}

extern "C" std::int32_t dev_finalize_metrics_summary(const DeviceState *state) {
  finalize_metrics_reduce_kernel<<<1U, 1U>>>(state->counters, state->metrics_reduce_scratch,
                                             state->metrics_summary);
  return moonai_gpu::synchronize_kernels();
}
