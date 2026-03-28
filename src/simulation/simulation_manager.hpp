#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "simulation/entity.hpp"
#include "simulation/food_store.hpp"
#include "simulation/spatial_grid.hpp"
#include "simulation/step_state.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace moonai {

class Registry;
class EvolutionManager;
namespace gpu {
class GpuBatch;
}

struct SimEvent {
  enum Type : uint8_t { Kill, Food, Birth, Death };
  Type type;
  Entity agent_id;  // predator (kill) or prey (food)
  Entity target_id; // prey (kill), food (food), or self (death)
  Vec2 position;    // where the event occurred
};

class SimulationManager {
public:
  struct ReproductionPair {
    Entity parent_a = INVALID_ENTITY;
    Entity parent_b = INVALID_ENTITY;
    Vec2 spawn_position;
  };

  struct SimulationStepResult {
    std::vector<SimEvent> events;
    std::vector<ReproductionPair> reproduction_pairs;
  };

  explicit SimulationManager(const SimulationConfig &config);
  ~SimulationManager();

  void initialize();
  SimulationStepResult step(Registry &registry, EvolutionManager &evolution);
  SimulationStepResult step_gpu(Registry &registry,
                                EvolutionManager &evolution);
  void reset();

  void enable_gpu(bool enable);
  bool gpu_enabled() const {
    return gpu_enabled_;
  }

  int current_step() const {
    return current_step_;
  }
  void increment_step() {
    ++current_step_;
  }

  SpatialGrid &spatial_grid() {
    return grid_;
  }
  const SpatialGrid &spatial_grid() const {
    return grid_;
  }

  const FoodStore &food_store() const {
    return food_store_;
  }

  int alive_predators() const {
    return alive_predators_;
  }
  int alive_prey() const {
    return alive_prey_;
  }

  void refresh_state(Registry &registry);

private:
  void initialize(bool log_initialization);

  PackedStepState pack_step_state(const Registry &registry) const;
  void apply_step_state(Registry &registry, const PackedStepState &state);
  void run_cpu_backend(PackedStepState &state, EvolutionManager &evolution);
  void collect_step_events(Registry &registry, const PackedStepState &state,
                           std::vector<SimEvent> &events);
  std::vector<ReproductionPair>
  find_reproduction_pairs(const Registry &registry) const;
  void refresh_world_state_after_step(Registry &registry);
  void rebuild_food_grid();
  void rebuild_spatial_grid(const Registry &registry);
  void count_alive(const Registry &registry);

  SimulationConfig config_;
  Random rng_;
  SpatialGrid grid_;
  SpatialGrid food_grid_;
  FoodStore food_store_;
  int current_step_ = 0;
  int alive_predators_ = 0;
  int alive_prey_ = 0;

  // GPU support
  bool gpu_enabled_ = false;
  std::unique_ptr<gpu::GpuBatch> gpu_batch_;
};

} // namespace moonai
