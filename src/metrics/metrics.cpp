#include "metrics/metrics.hpp"

#include "core/app_state.hpp"

#include <algorithm>
#include <cstdint>

namespace moonai::metrics {

namespace {

int count_active_food(const Food &food) {
  int active_food = 0;
  for (uint8_t active : food.active) {
    active_food += active ? 1 : 0;
  }
  return active_food;
}

} // namespace

void refresh(AppState &state) {
  MetricsSnapshot &metrics = state.metrics;
  metrics.step = state.runtime.step;
  metrics.predator_count = static_cast<int>(state.predator.size());
  metrics.prey_count = static_cast<int>(state.prey.size());
  metrics.active_food = count_active_food(state.food);
  metrics.predator_species = static_cast<int>(state.predator.species.size());
  metrics.prey_species = static_cast<int>(state.prey.species.size());

  // Calculate average energy for predators
  float predator_energy_sum = 0.0f;
  for (float energy : state.predator.energy) {
    predator_energy_sum += energy;
  }
  metrics.avg_predator_energy = 0.0f;
  if (!state.predator.energy.empty()) {
    metrics.avg_predator_energy = predator_energy_sum / static_cast<float>(state.predator.energy.size());
  }

  // Calculate average energy for prey
  float prey_energy_sum = 0.0f;
  for (float energy : state.prey.energy) {
    prey_energy_sum += energy;
  }
  metrics.avg_prey_energy = 0.0f;
  if (!state.prey.energy.empty()) {
    metrics.avg_prey_energy = prey_energy_sum / static_cast<float>(state.prey.energy.size());
  }

  // Calculate average complexity for predators
  float predator_complexity_sum = 0.0f;
  for (const auto &genome : state.predator.genomes) {
    predator_complexity_sum += static_cast<float>(genome.nodes().size() + genome.connections().size());
  }
  metrics.avg_predator_complexity = 0.0f;
  if (!state.predator.genomes.empty()) {
    metrics.avg_predator_complexity = predator_complexity_sum / static_cast<float>(state.predator.genomes.size());
  }

  // Calculate average complexity for prey
  float prey_complexity_sum = 0.0f;
  for (const auto &genome : state.prey.genomes) {
    prey_complexity_sum += static_cast<float>(genome.nodes().size() + genome.connections().size());
  }
  metrics.avg_prey_complexity = 0.0f;
  if (!state.prey.genomes.empty()) {
    metrics.avg_prey_complexity = prey_complexity_sum / static_cast<float>(state.prey.genomes.size());
  }

  // Calculate generation metrics for predators
  int max_pred_gen = 0;
  int64_t pred_gen_sum = 0;
  for (int gen : state.predator.generation) {
    max_pred_gen = std::max(max_pred_gen, gen);
    pred_gen_sum += gen;
  }
  metrics.max_predator_generation = 0;
  metrics.avg_predator_generation = 0.0f;
  if (!state.predator.generation.empty()) {
    metrics.max_predator_generation = max_pred_gen;
    metrics.avg_predator_generation =
        static_cast<float>(pred_gen_sum) / static_cast<float>(state.predator.generation.size());
  }

  // Calculate generation metrics for prey
  int max_prey_gen = 0;
  int64_t prey_gen_sum = 0;
  for (int gen : state.prey.generation) {
    max_prey_gen = std::max(max_prey_gen, gen);
    prey_gen_sum += gen;
  }
  metrics.max_prey_generation = 0;
  metrics.avg_prey_generation = 0.0f;
  if (!state.prey.generation.empty()) {
    metrics.max_prey_generation = max_prey_gen;
    metrics.avg_prey_generation = static_cast<float>(prey_gen_sum) / static_cast<float>(state.prey.generation.size());
  }
}

} // namespace moonai::metrics
