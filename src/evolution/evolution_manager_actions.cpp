#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "simulation/simulation_step_systems.hpp"

namespace moonai {

using simulation_detail::OUTPUT_COUNT;
using simulation_detail::SENSOR_COUNT;

void EvolutionManager::compute_actions_for_population(
    PopulationEvolutionState &population, AgentRegistry &agents,
    const std::vector<float> &sensors,
    std::vector<float> &decisions_out) const {
  const uint32_t entity_count = static_cast<uint32_t>(agents.size());

  std::vector<float> all_outputs;
  population.network_cache.activate_batch(entity_count, sensors, all_outputs,
                                          SENSOR_COUNT, OUTPUT_COUNT);

  decisions_out.resize(entity_count * OUTPUT_COUNT);
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    decisions_out[idx * OUTPUT_COUNT] = all_outputs[idx * OUTPUT_COUNT];
    decisions_out[idx * OUTPUT_COUNT + 1] = all_outputs[idx * OUTPUT_COUNT + 1];
  }
}

void EvolutionManager::compute_actions(
    AppState &state, const std::vector<float> &predator_sensors,
    const std::vector<float> &prey_sensors,
    std::vector<float> &predator_decisions,
    std::vector<float> &prey_decisions) {
  MOONAI_PROFILE_SCOPE("evolution_compute_actions");
  compute_actions_for_population(state.evolution.predators, state.predators,
                                 predator_sensors, predator_decisions);
  compute_actions_for_population(state.evolution.prey, state.prey, prey_sensors,
                                 prey_decisions);
}

} // namespace moonai