#include "simulation/simulation_manager.hpp"

#include "core/app_state.hpp"
#include "gpu/gpu_batch.hpp"

#include <spdlog/spdlog.h>

namespace moonai {

SimulationManager::SimulationManager(const SimulationConfig &config) : config_(config) {}

SimulationManager::~SimulationManager() = default;

void SimulationManager::initialize(AppState &state) {
  initialize(state, true);
}

void SimulationManager::initialize(AppState &state, bool log_initialization) {
  state.food.initialize(config_, state.runtime.rng);

  if (log_initialization) {
    spdlog::info("Simulation initialized: {} food pellets (seed: {})", config_.food_count, state.runtime.rng.seed());
  }
}

void SimulationManager::reset(AppState &state) {
  initialize(state, false);
}

} // namespace moonai
