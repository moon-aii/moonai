#pragma once
#include "core/types.hpp"
#include "simulation/spatial_grid_ecs.hpp"
#include "simulation/system.hpp"
#include <vector>

namespace moonai {

class CombatSystem : public System {
public:
  struct KillEvent {
    Entity killer;
    Entity victim;
  };

  struct FoodEvent {
    Entity eater;
    size_t food_index;
  };

  CombatSystem(const SpatialGridECS &agent_grid, float attack_range);

  void update(Registry &registry) override;
  const char *name() const override {
    return "CombatSystem";
  }

  const std::vector<KillEvent> &kill_events() const {
    return kill_events_;
  }
  const std::vector<FoodEvent> &food_events() const {
    return food_events_;
  }

  void clear_events() {
    kill_events_.clear();
    food_events_.clear();
  }

private:
  const SpatialGridECS &agent_grid_;
  float attack_range_;
  float attack_range_sq_;

  std::vector<KillEvent> kill_events_;
  std::vector<FoodEvent> food_events_;

  void process_predator_attacks(Registry &registry);
};

} // namespace moonai
