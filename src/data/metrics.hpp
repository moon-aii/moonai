#pragma once

#include "evolution/genome.hpp"

#include <vector>

namespace moonai {

class Registry;
class EvolutionManager;

struct StepMetrics {
  int step = 0;
  int predator_count = 0;
  int prey_count = 0;
  int births = 0;
  int deaths = 0;
  float best_fitness = 0.0f;
  float avg_fitness = 0.0f;
  int num_species = 0;
  float avg_genome_complexity = 0.0f;
  float avg_predator_energy = 0.0f;
  float avg_prey_energy = 0.0f;
};

class MetricsCollector {
public:
  StepMetrics collect_ecs(int step, const Registry &registry,
                          const EvolutionManager &evolution, int births,
                          int deaths, int num_species);

  const std::vector<StepMetrics> &history() const {
    return history_;
  }

private:
  std::vector<StepMetrics> history_;
};

} // namespace moonai
