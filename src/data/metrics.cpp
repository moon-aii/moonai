#include "data/metrics.hpp"

#include "core/app_state.hpp"

namespace moonai::metrics {

namespace {

int count_active_food(const FoodStore &food_store) {
  int active_food = 0;
  for (uint8_t active : food_store.active()) {
    active_food += active ? 1 : 0;
  }
  return active_food;
}

} // namespace

void refresh_live(AppState &state) {
  state.metrics.live.alive_predators = 0;
  state.metrics.live.alive_prey = 0;
  state.metrics.live.active_food = count_active_food(state.food_store);
  state.metrics.live.num_species =
      static_cast<int>(state.evolution.species.size());

  const auto &identity = state.registry.identity();
  for (std::size_t idx = 0; idx < state.registry.size(); ++idx) {
    if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
      ++state.metrics.live.alive_predators;
    } else {
      ++state.metrics.live.alive_prey;
    }
  }
}

void record_report(AppState &state) {
  refresh_live(state);

  ReportMetrics report;
  report.step = state.runtime.step;
  report.predator_count = state.metrics.live.alive_predators;
  report.prey_count = state.metrics.live.alive_prey;
  report.births = state.runtime.report_events.births;
  report.deaths = state.runtime.report_events.deaths;
  report.num_species = state.metrics.live.num_species;

  float predator_energy_sum = 0.0f;
  float prey_energy_sum = 0.0f;
  int predator_energy_count = 0;
  int prey_energy_count = 0;
  float complexity_sum = 0.0f;
  int genome_count = 0;

  const auto &vitals = state.registry.vitals();
  const auto &identity = state.registry.identity();
  for (std::size_t idx = 0; idx < state.registry.size(); ++idx) {
    if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
      predator_energy_sum += vitals.energy[idx];
      ++predator_energy_count;
    } else {
      prey_energy_sum += vitals.energy[idx];
      ++prey_energy_count;
    }

    if (idx < state.evolution.entity_genomes.size()) {
      const auto &genome = state.evolution.entity_genomes[idx];
      complexity_sum += static_cast<float>(genome.nodes().size() +
                                           genome.connections().size());
      ++genome_count;
    }
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
