#pragma once
#include "core/types.hpp"
#include "simulation/system.hpp"

namespace moonai {

// Manages food respawning using deterministic random
// Food entities are recycled: inactive food slots are respawned each step
class FoodRespawnSystem : public System {
public:
  FoodRespawnSystem(float world_width, float world_height, float respawn_rate,
                    std::uint64_t seed);

  void update(Registry &registry, float dt) override;
  const char *name() const override {
    return "FoodRespawnSystem";
  }

  // Call before each step to set the current step index for deterministic
  // respawn
  void set_step(int step_index) {
    step_index_ = step_index;
  }

private:
  float world_width_;
  float world_height_;
  float respawn_rate_;
  std::uint64_t seed_;
  int step_index_ = 0;

  // Food visual properties (constant)
  static constexpr float FOOD_RADIUS = 3.0f;
  static constexpr uint32_t FOOD_COLOR = 0x00FF00FF; // Green
};

} // namespace moonai