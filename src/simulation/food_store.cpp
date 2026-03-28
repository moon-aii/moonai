#include "simulation/food_store.hpp"

#include "core/deterministic_respawn.hpp"

namespace moonai {

void FoodStore::initialize(const SimulationConfig &config, Random &rng) {
  pos_x_.resize(config.food_count);
  pos_y_.resize(config.food_count);
  active_.assign(config.food_count, 1);
  slot_index_.resize(config.food_count);

  const float grid_size = static_cast<float>(config.grid_size);
  for (int i = 0; i < config.food_count; ++i) {
    pos_x_[static_cast<std::size_t>(i)] = rng.next_float(0.0f, grid_size);
    pos_y_[static_cast<std::size_t>(i)] = rng.next_float(0.0f, grid_size);
    slot_index_[static_cast<std::size_t>(i)] = static_cast<uint32_t>(i);
  }
}

void FoodStore::respawn_step(const SimulationConfig &config, int step_index,
                             std::uint64_t seed) {
  const float world_width = static_cast<float>(config.grid_size);
  const float world_height = static_cast<float>(config.grid_size);

  for (std::size_t i = 0; i < active_.size(); ++i) {
    if (active_[i]) {
      continue;
    }

    const uint32_t slot = slot_index_[i];
    if (!respawn::should_respawn(seed, step_index, slot,
                                 config.food_respawn_rate)) {
      continue;
    }

    pos_x_[i] = respawn::respawn_x(seed, step_index, slot, world_width);
    pos_y_[i] = respawn::respawn_y(seed, step_index, slot, world_height);
    active_[i] = 1;
  }
}

} // namespace moonai
