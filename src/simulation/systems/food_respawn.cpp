#include "simulation/systems/food_respawn.hpp"
#include "core/deterministic_respawn.hpp"
#include "simulation/components.hpp"

namespace moonai {

FoodRespawnSystem::FoodRespawnSystem(float world_width, float world_height,
                                     float respawn_rate, std::uint64_t seed)
    : world_width_(world_width), world_height_(world_height),
      respawn_rate_(respawn_rate), seed_(seed) {}

void FoodRespawnSystem::update(Registry &registry, float dt) {
  (void)dt; // dt not used for food respawn

  auto &living = registry.living_entities();
  auto &positions = registry.positions();
  auto &food_state = registry.food_state();
  auto &identity = registry.identity();

  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);

    // Only process food entities
    if (identity.type[idx] != IdentitySoA::TYPE_FOOD) {
      continue;
    }

    // Skip already active food
    if (food_state.active[idx]) {
      continue;
    }

    // Check if this food slot should respawn
    uint32_t slot_index = food_state.slot_index[idx];
    if (respawn::should_respawn(seed_, step_index_, slot_index,
                                respawn_rate_)) {
      // Respawn this food
      positions.x[idx] =
          respawn::respawn_x(seed_, step_index_, slot_index, world_width_);
      positions.y[idx] =
          respawn::respawn_y(seed_, step_index_, slot_index, world_height_);
      food_state.active[idx] = 1;
    }
  }
}

} // namespace moonai