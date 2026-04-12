#include "simulation/simulation.hpp"

#include "simulation/common.hpp"
#include "simulation/cpu.hpp"
#include "simulation/gpu.hpp"

namespace moonai::simulation {

void initialize(AppState &state, const SimulationConfig &config) {
  state.food.initialize(config, state.runtime.rng);
}

bool prepare_step(AppState &state, const SimulationConfig &config) {
#ifdef MOONAI_ENABLE_CUDA
  if (state.runtime.gpu_enabled) {
    return gpu::prepare_step(state, config);
  }
#endif

  return cpu::prepare_step(state, config);
}

bool resolve_step(AppState &state, const SimulationConfig &config) {
#ifdef MOONAI_ENABLE_CUDA
  if (state.runtime.gpu_enabled) {
    return gpu::resolve_step(state, config);
  }
#endif

  return cpu::resolve_step(state, config);
}

void post_step(AppState &state, const SimulationConfig &config) {
  common::post_step(state, config);
}

} // namespace moonai::simulation
