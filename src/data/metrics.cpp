#include "data/metrics.hpp"

#include "evolution/evolution_manager.hpp"
#include "simulation/registry.hpp"
#include "simulation/simulation_manager.hpp"

#include <algorithm>
#include <numeric>

namespace moonai {

StepMetrics MetricsCollector::collect_ecs(int step, const Registry &registry,
                                          const EvolutionManager &evolution,
                                          const std::vector<SimEvent> &events,
                                          int num_species) {
  StepMetrics metrics;
  metrics.step = step;
  metrics.num_species = num_species;

  // Count births and deaths from events
  int births = 0;
  int deaths = 0;
  for (const auto &event : events) {
    if (event.type == SimEvent::Birth) {
      ++births;
    } else if (event.type == SimEvent::Death) {
      ++deaths;
    }
  }
  metrics.births = births;
  metrics.deaths = deaths;

  float predator_energy_sum = 0.0f;
  float prey_energy_sum = 0.0f;
  int predator_energy_count = 0;
  int prey_energy_count = 0;

  const auto &living = registry.living_entities();
  const auto &vitals = registry.vitals();
  const auto &identity = registry.identity();

  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    if (!vitals.alive[idx]) {
      continue;
    }

    if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
      ++metrics.predator_count;
      predator_energy_sum += vitals.energy[idx];
      ++predator_energy_count;
    } else {
      ++metrics.prey_count;
      prey_energy_sum += vitals.energy[idx];
      ++prey_energy_count;
    }
  }

  if (predator_energy_count > 0) {
    metrics.avg_predator_energy =
        predator_energy_sum / static_cast<float>(predator_energy_count);
  }
  if (prey_energy_count > 0) {
    metrics.avg_prey_energy =
        prey_energy_sum / static_cast<float>(prey_energy_count);
  }

  std::vector<float> fitnesses;
  float complexity_sum = 0.0f;
  int genome_count = 0;

  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    if (!vitals.alive[idx]) {
      continue;
    }

    const Genome *genome = evolution.genome_for(entity);
    if (genome) {
      fitnesses.push_back(genome->fitness());
      complexity_sum += static_cast<float>(genome->complexity());
      ++genome_count;
    }
  }

  if (!fitnesses.empty()) {
    metrics.best_fitness =
        *std::max_element(fitnesses.begin(), fitnesses.end());

    const float fitness_sum =
        std::accumulate(fitnesses.begin(), fitnesses.end(), 0.0f);
    metrics.avg_fitness = fitness_sum / static_cast<float>(fitnesses.size());

    metrics.avg_genome_complexity =
        complexity_sum / static_cast<float>(genome_count);
  }

  history_.push_back(metrics);
  return metrics;
}

} // namespace moonai
