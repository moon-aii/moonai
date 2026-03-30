#pragma once

#include "core/config.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace moonai {

struct AppState;
struct PendingOffspring;
class EvolutionManager;
namespace gpu {
class GpuBatch;
}

class SimulationManager {
public:
  explicit SimulationManager(const SimulationConfig &config);
  ~SimulationManager();

  void initialize(AppState &state);
  void step(AppState &state, EvolutionManager &evolution);
  void step_gpu(AppState &state, EvolutionManager &evolution);
  void reset(AppState &state);

  void enable_gpu(bool enable);
  bool gpu_enabled() const {
    return gpu_enabled_;
  }

private:
  void initialize(AppState &state, bool log_initialization);
  void ensure_gpu_capacity(std::size_t predator_count, std::size_t prey_count,
                           std::size_t food_count);

  void compact_predators(AppState &state, EvolutionManager &evolution);
  void compact_prey(AppState &state, EvolutionManager &evolution);
  void collect_gpu_step_events(AppState &state,
                               const std::vector<uint8_t> &was_predator_alive,
                               const std::vector<uint8_t> &was_prey_alive,
                               const std::vector<uint8_t> &was_food_active);
  std::vector<PendingOffspring>
  find_predator_reproduction_pairs(const AppState &state) const;
  std::vector<PendingOffspring>
  find_prey_reproduction_pairs(const AppState &state) const;
  void refresh_world_state_after_step(AppState &state);

  const SimulationConfig &config_;

  // GPU support
  bool gpu_enabled_ = false;
  std::unique_ptr<gpu::GpuBatch> gpu_batch_;
};

} // namespace moonai
