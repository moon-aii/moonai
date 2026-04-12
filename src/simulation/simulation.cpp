#include "simulation/simulation.hpp"

#include "simulation/common.hpp"
#include "simulation/gpu.hpp"

namespace moonai::simulation {

void initialize(AppState &state, const SimulationConfig &config) {
  state.food.initialize(config, state.runtime.rng);
}

bool prepare_step(AppState &state, const SimulationConfig &config) {
  return gpu::prepare_step(state, config);
}

bool resolve_step(AppState &state, const SimulationConfig &config) {
  return gpu::resolve_step(state, config);
}

void post_step(AppState &state, const SimulationConfig &config) {
  common::post_step(state, config);
}

} // namespace moonai::simulation
