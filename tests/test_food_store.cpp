#include "simulation/food_store.hpp"

#include "core/config.hpp"
#include "core/random.hpp"

#include <gtest/gtest.h>

using namespace moonai;

TEST(FoodStoreTest, InitializesConfiguredFoodCount) {
  SimulationConfig config;
  config.food_count = 8;
  config.grid_size = 100;

  Random rng(42);
  FoodStore food_store;
  food_store.initialize(config, rng);

  EXPECT_EQ(food_store.size(), 8u);
  for (std::size_t i = 0; i < food_store.size(); ++i) {
    EXPECT_TRUE(food_store.active()[i]);
    EXPECT_GE(food_store.pos_x()[i], 0.0f);
    EXPECT_LT(food_store.pos_x()[i], 100.0f);
    EXPECT_GE(food_store.pos_y()[i], 0.0f);
    EXPECT_LT(food_store.pos_y()[i], 100.0f);
    EXPECT_EQ(food_store.slot_index()[i], i);
  }
}

TEST(FoodStoreTest, RespawnsInactiveFoodDeterministically) {
  SimulationConfig config;
  config.food_count = 4;
  config.grid_size = 200;
  config.food_respawn_rate = 1.0f;

  Random rng(7);
  FoodStore food_store;
  food_store.initialize(config, rng);

  food_store.active()[2] = 0;
  const float old_x = food_store.pos_x()[2];
  const float old_y = food_store.pos_y()[2];

  food_store.respawn_step(config, 5, 7);

  EXPECT_TRUE(food_store.active()[2]);
  EXPECT_NE(food_store.pos_x()[2], old_x);
  EXPECT_NE(food_store.pos_y()[2], old_y);
}
