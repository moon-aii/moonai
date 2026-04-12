#include "simulation/manager.hpp"

#include "core/app_state.hpp"
#include "evolution/evolution_manager.hpp"
#include "gpu/gpu_batch.hpp"
#include "simulation/common.hpp"
#include "simulation/cpu.hpp"
#include "simulation/gpu.hpp"

#include <spdlog/spdlog.h>

namespace moonai {

SimulationManager::SimulationManager(const SimulationConfig &config) : config_(config) {}

SimulationManager::~SimulationManager() = default;

void SimulationManager::initialize(AppState &state) {
  state.food.initialize(config_, state.runtime.rng);
}

void SimulationManager::step(AppState &state, EvolutionManager &evolution) {
  if (state.runtime.gpu_enabled) {
    gpu::step(state, evolution, gpu_batch_, config_);
  } else {
    cpu::step(state, evolution, config_);
  }

  common::run(state, evolution, config_);
}

void SimulationManager::enable_gpu(AppState &state, bool enable) {
  if (enable) {
    gpu::ensure_capacity(gpu_batch_, static_cast<std::size_t>(config_.predator_count),
                         static_cast<std::size_t>(config_.prey_count), static_cast<std::size_t>(config_.food_count));
    state.runtime.gpu_enabled = true;
  } else {
    disable_gpu(state);
  }
}

void SimulationManager::disable_gpu(AppState &state) {
  gpu::disable(gpu_batch_);
  state.runtime.gpu_enabled = false;
}

} // namespace moonai
