#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"

namespace moonai {

void EvolutionManager::compute_actions(AppState &state) {
  MOONAI_PROFILE_SCOPE("evolution_compute_actions");
  const uint32_t entity_count = static_cast<uint32_t>(state.registry.size());

  std::vector<float> all_inputs;
  all_inputs.reserve(entity_count * SensorSoA::INPUT_COUNT);

  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    const float *input_ptr = state.registry.sensors.input_ptr(idx);
    all_inputs.insert(all_inputs.end(), input_ptr,
                      input_ptr + SensorSoA::INPUT_COUNT);
  }

  std::vector<float> all_outputs;
  compute_actions_batch(entity_count, state, all_inputs, all_outputs);

  for (uint32_t i = 0; i < entity_count; ++i) {
    state.registry.brain.decision_x[i] = all_outputs[i * 2];
    state.registry.brain.decision_y[i] = all_outputs[i * 2 + 1];
  }
}

void EvolutionManager::compute_actions_batch(
    std::size_t entity_count, AppState &state,
    const std::vector<float> &all_inputs, std::vector<float> &all_outputs) {
  state.evolution.network_cache.activate_batch(
      entity_count, all_inputs, all_outputs, SensorSoA::INPUT_COUNT,
      SensorSoA::OUTPUT_COUNT);
}

} // namespace moonai
