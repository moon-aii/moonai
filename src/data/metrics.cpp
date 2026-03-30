#include "data/metrics.hpp"

#include "core/app_state.hpp"

namespace moonai::metrics {

namespace {

int count_active_food(const FoodStore &food_store) {
  int active_food = 0;
  for (uint8_t active : food_store.active) {
    active_food += active ? 1 : 0;
  }
  return active_food;
}

} // namespace

void refresh_live(AppState &state) {
  state.metrics.live.alive_predators = static_cast<int>(state.predators.size());
  state.metrics.live.alive_prey = static_cast<int>(state.prey.size());
  state.metrics.live.active_food = count_active_food(state.food_store);
  state.metrics.live.predator_species =
      static_cast<int>(state.evolution.predators.species.size());
  state.metrics.live.prey_species =
      static_cast<int>(state.evolution.prey.species.size());
}

void record_report(AppState &state) {
  refresh_live(state);

  ReportMetrics report;
  report.step = state.runtime.step;
  report.predator_count = state.metrics.live.alive_predators;
  report.prey_count = state.metrics.live.alive_prey;
  report.births = state.runtime.report_events.births;
  report.deaths = state.runtime.report_events.deaths;
  report.predator_species = state.metrics.live.predator_species;
  report.prey_species = state.metrics.live.prey_species;

  float predator_energy_sum = 0.0f;
  float prey_energy_sum = 0.0f;
  int predator_energy_count = 0;
  int prey_energy_count = 0;
  float complexity_sum = 0.0f;
  int genome_count = 0;

  for (float energy : state.predators.agents.energy) {
    predator_energy_sum += energy;
    ++predator_energy_count;
  }

  for (float energy : state.prey.agents.energy) {
    prey_energy_sum += energy;
    ++prey_energy_count;
  }

  for (const auto &genome : state.evolution.predators.genomes) {
    complexity_sum +=
        static_cast<float>(genome.nodes().size() + genome.connections().size());
    ++genome_count;
  }

  for (const auto &genome : state.evolution.prey.genomes) {
    complexity_sum +=
        static_cast<float>(genome.nodes().size() + genome.connections().size());
    ++genome_count;
  }

  if (predator_energy_count > 0) {
    report.avg_predator_energy =
        predator_energy_sum / static_cast<float>(predator_energy_count);
  }
  if (prey_energy_count > 0) {
    report.avg_prey_energy =
        prey_energy_sum / static_cast<float>(prey_energy_count);
  }
  if (genome_count > 0) {
    report.avg_genome_complexity =
        complexity_sum / static_cast<float>(genome_count);
  }

  state.metrics.last_report = report;
  state.metrics.history.push_back(report);
}

} // namespace moonai::metrics
