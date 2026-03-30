#include "simulation/food_store.hpp"

#include "core/deterministic_respawn.hpp"

namespace moonai {

void FoodStore::initialize(const SimulationConfig &config, Random &rng) {
  pos_x.resize(config.food_count);
  pos_y.resize(config.food_count);
  active.assign(config.food_count, 1);

  const float grid_size = static_cast<float>(config.grid_size);
  for (int i = 0; i < config.food_count; ++i) {
    pos_x[static_cast<std::size_t>(i)] = rng.next_float(0.0f, grid_size);
    pos_y[static_cast<std::size_t>(i)] = rng.next_float(0.0f, grid_size);
  }
}

void FoodStore::respawn_step(const SimulationConfig &config, int step_index,
                             std::uint64_t seed) {
  const float world_size = static_cast<float>(config.grid_size);

  for (std::size_t i = 0; i < active.size(); ++i) {
    if (active[i]) {
      continue;
    }

    const uint32_t slot = static_cast<uint32_t>(i);
    if (!respawn::should_respawn(seed, step_index, slot,
                                 config.food_respawn_rate)) {
      continue;
    }

    pos_x[i] = respawn::respawn_x(seed, step_index, slot, world_size);
    pos_y[i] = respawn::respawn_y(seed, step_index, slot, world_size);
    active[i] = 1;
  }
}

} // namespace moonai
