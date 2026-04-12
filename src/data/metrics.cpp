#include "data/metrics.hpp"

#include "core/app_state.hpp"

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

void begin_step(AppState &state) {
  state.metrics.step_delta.clear();
}

void refresh_live(AppState &state) {
  MetricsSnapshot &live = state.metrics.live;
  live.clear();
  live.step = state.runtime.step;
  live.predator_count = static_cast<int>(state.predator.size());
  live.prey_count = static_cast<int>(state.prey.size());
  live.active_food = count_active_food(state.food);
  live.predator_species = static_cast<int>(state.predator.species.size());
  live.prey_species = static_cast<int>(state.prey.species.size());
}

void finalize_step(AppState &state) {
  refresh_live(state);
  state.metrics.report_window.step = state.metrics.live.step;
  state.metrics.report_window.add_events(state.metrics.step_delta);
  state.metrics.totals.step = state.metrics.live.step;
  state.metrics.totals.add_events(state.metrics.step_delta);
}

void record_report(AppState &state) {
  MetricsSnapshot report;
  report.step = state.metrics.live.step;
  report.predator_count = state.metrics.live.predator_count;
  report.prey_count = state.metrics.live.prey_count;
  report.active_food = state.metrics.live.active_food;
  report.kills = state.metrics.report_window.kills;
  report.food_eaten = state.metrics.report_window.food_eaten;
  report.births = state.metrics.report_window.births;
  report.deaths = state.metrics.report_window.deaths;
  report.predator_species = state.metrics.live.predator_species;
  report.prey_species = state.metrics.live.prey_species;

  float predator_energy_sum = 0.0f;
  float prey_energy_sum = 0.0f;
  int predator_energy_count = 0;
  int prey_energy_count = 0;
  float complexity_sum = 0.0f;
  int genome_count = 0;

  for (float energy : state.predator.energy) {
    predator_energy_sum += energy;
    ++predator_energy_count;
  }

  for (float energy : state.prey.energy) {
    prey_energy_sum += energy;
    ++prey_energy_count;
  }

  for (const auto &genome : state.predator.genomes) {
    complexity_sum += static_cast<float>(genome.nodes().size() + genome.connections().size());
    ++genome_count;
  }

  for (const auto &genome : state.prey.genomes) {
    complexity_sum += static_cast<float>(genome.nodes().size() + genome.connections().size());
    ++genome_count;
  }

  if (predator_energy_count > 0) {
    report.avg_predator_energy = predator_energy_sum / static_cast<float>(predator_energy_count);
  }
  if (prey_energy_count > 0) {
    report.avg_prey_energy = prey_energy_sum / static_cast<float>(prey_energy_count);
  }
  if (genome_count > 0) {
    report.avg_genome_complexity = complexity_sum / static_cast<float>(genome_count);
  }

  state.metrics.last_report = report;
  state.metrics.has_last_report = true;
  state.metrics.report_window.clear_events();
}

} // namespace moonai::metrics
