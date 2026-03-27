#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "simulation/entity.hpp"
#include "simulation/spatial_grid_ecs.hpp"
#include "simulation/systems/combat.hpp"
#include "simulation/systems/energy.hpp"
#include "simulation/systems/food_respawn.hpp"
#include "simulation/systems/movement.hpp"
#include "simulation/systems/sensor.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace moonai {

class Registry;

struct SimEvent {
  enum Type : uint8_t { Kill, Food, Birth, Death };
  Type type;
  Entity agent_id;  // predator (kill) or prey (food)
  Entity target_id; // prey (kill), food (food), or self (death)
  Entity parent_a_id = INVALID_ENTITY;
  Entity parent_b_id = INVALID_ENTITY;
  Vec2 position; // where the event occurred
};

class SimulationManager {
public:
  explicit SimulationManager(const SimulationConfig &config);

  void initialize();
  void step_ecs(Registry &registry, float dt);
  void reset();

  int current_step() const {
    return current_step_;
  }
  void increment_step() {
    ++current_step_;
  }

  SpatialGridECS &spatial_grid() {
    return grid_;
  }
  const SpatialGridECS &spatial_grid() const {
    return grid_;
  }

  int alive_predators() const {
    return alive_predators_;
  }
  int alive_prey() const {
    return alive_prey_;
  }

  const std::vector<SimEvent> &last_events() const {
    return last_events_;
  }
  void record_event(const SimEvent &event) {
    last_events_.push_back(event);
  }

  struct ReproductionPair {
    Entity parent_a = INVALID_ENTITY;
    Entity parent_b = INVALID_ENTITY;
    Vec2 spawn_position;
  };

  std::vector<ReproductionPair>
  find_reproduction_pairs_ecs(const Registry &registry) const;

  void refresh_state_ecs(Registry &registry);

private:
  void initialize(bool log_initialization);

  void rebuild_spatial_grid_ecs(const Registry &registry);
  void process_food_ecs(Registry &registry);
  void process_step_deaths_ecs(Registry &registry);
  void count_alive_ecs(const Registry &registry);

  SimulationConfig config_;
  Random rng_;
  SpatialGridECS grid_;
  std::vector<SimEvent> last_events_;
  int current_step_ = 0;
  int alive_predators_ = 0;
  int alive_prey_ = 0;

  SensorSystem sensor_system_;
  EnergySystem energy_system_;
  MovementSystem movement_system_;
  CombatSystem combat_system_;
  FoodRespawnSystem food_respawn_system_;
};

} // namespace moonai
