#include "core/app_state.hpp"
#include "core/random.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/registry.hpp"
#include "simulation/simulation_manager.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace moonai {
namespace {

constexpr float kEpsilon = 1e-5f;
constexpr float kMissingTargetSentinel = 2.0f;
constexpr float kMaxDensity = 10.0f;

float distance_sq(Vec2 diff) {
  return diff.x * diff.x + diff.y * diff.y;
}

TEST(SimulationSensorsTest, EncodesDxDySentinelsAndFoodDensity) {
  SimulationConfig config;
  config.grid_size = 100;
  config.seed = 7;
  config.predator_count = 1;
  config.prey_count = 1;
  config.food_count = 2;
  config.vision_range = 100.0f;
  config.predator_speed = 0.0f;
  config.prey_speed = 0.0f;
  config.energy_drain_per_step = 0.0f;
  config.interaction_range = 0.0f;

  AppState state(config.seed);
  SimulationManager simulation(config);
  simulation.initialize(state);

  EvolutionManager evolution(config);
  evolution.initialize(state, AgentRegistry::INPUT_COUNT,
                       AgentRegistry::OUTPUT_COUNT);
  evolution.seed_initial_population(state);

  ASSERT_EQ(state.predators.size(), 1u);
  ASSERT_EQ(state.prey.size(), 1u);
  ASSERT_EQ(AgentRegistry::INPUT_COUNT, 12);

  const std::size_t predator_idx = 0;
  const std::size_t prey_idx = 0;
  state.predators.pos_x[predator_idx] = 10.0f;
  state.predators.pos_y[predator_idx] = 10.0f;
  state.prey.pos_x[prey_idx] = 22.0f;
  state.prey.pos_y[prey_idx] = 16.0f;

  simulation.step(state, evolution);

  const auto &food_store = state.food_store;
  ASSERT_EQ(food_store.size(), 2u);
  ASSERT_TRUE(food_store.active[0]);
  ASSERT_TRUE(food_store.active[1]);

  auto nearest_food_delta = [&](const Vec2 &origin) {
    Vec2 best{0.0f, 0.0f};
    float best_dist = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < food_store.size(); ++i) {
      Vec2 diff{food_store.pos_x[i] - origin.x, food_store.pos_y[i] - origin.y};
      const float dist = distance_sq(diff);
      if (dist < best_dist) {
        best_dist = dist;
        best = diff;
      }
    }
    return best;
  };

  const Vec2 predator_pos{state.predators.pos_x[predator_idx],
                          state.predators.pos_y[predator_idx]};
  const Vec2 prey_pos{state.prey.pos_x[prey_idx], state.prey.pos_y[prey_idx]};
  const Vec2 predator_to_prey{prey_pos.x - predator_pos.x,
                              prey_pos.y - predator_pos.y};
  const Vec2 prey_to_predator{predator_pos.x - prey_pos.x,
                              predator_pos.y - prey_pos.y};
  const Vec2 predator_to_food = nearest_food_delta(predator_pos);
  const Vec2 prey_to_food = nearest_food_delta(prey_pos);

  const float *predator_sensors = state.predators.input_ptr(predator_idx);
  EXPECT_FLOAT_EQ(predator_sensors[0], kMissingTargetSentinel);
  EXPECT_FLOAT_EQ(predator_sensors[1], kMissingTargetSentinel);
  EXPECT_NEAR(predator_sensors[2], predator_to_prey.x / config.vision_range,
              kEpsilon);
  EXPECT_NEAR(predator_sensors[3], predator_to_prey.y / config.vision_range,
              kEpsilon);
  EXPECT_NEAR(predator_sensors[4], predator_to_food.x / config.vision_range,
              kEpsilon);
  EXPECT_NEAR(predator_sensors[5], predator_to_food.y / config.vision_range,
              kEpsilon);
  EXPECT_FLOAT_EQ(predator_sensors[6], 0.5f);
  EXPECT_FLOAT_EQ(predator_sensors[7], 0.0f);
  EXPECT_FLOAT_EQ(predator_sensors[8], 0.0f);
  EXPECT_FLOAT_EQ(predator_sensors[9], 0.0f);
  EXPECT_FLOAT_EQ(predator_sensors[10], 1.0f / kMaxDensity);
  // With bounded world (no wrapping), food visibility depends on actual
  // distance Prior value assumed wrapping behavior:
  // EXPECT_FLOAT_EQ(predator_sensors[11], 2.0f / kMaxDensity);
  EXPECT_FLOAT_EQ(predator_sensors[11], 1.0f / kMaxDensity);

  const float *prey_sensors = state.prey.input_ptr(prey_idx);
  EXPECT_NEAR(prey_sensors[0], prey_to_predator.x / config.vision_range,
              kEpsilon);
  EXPECT_NEAR(prey_sensors[1], prey_to_predator.y / config.vision_range,
              kEpsilon);
  EXPECT_FLOAT_EQ(prey_sensors[2], kMissingTargetSentinel);
  EXPECT_FLOAT_EQ(prey_sensors[3], kMissingTargetSentinel);
  EXPECT_NEAR(prey_sensors[4], prey_to_food.x / config.vision_range, kEpsilon);
  EXPECT_NEAR(prey_sensors[5], prey_to_food.y / config.vision_range, kEpsilon);
  EXPECT_FLOAT_EQ(prey_sensors[6], 0.5f);
  EXPECT_FLOAT_EQ(prey_sensors[7], 0.0f);
  EXPECT_FLOAT_EQ(prey_sensors[8], 0.0f);
  EXPECT_FLOAT_EQ(prey_sensors[9], 1.0f / kMaxDensity);
  EXPECT_FLOAT_EQ(prey_sensors[10], 0.0f);
  // With bounded world (no wrapping), food visibility depends on actual
  // positions Prey at (22, 16) can see both food items within vision_range=100
  EXPECT_FLOAT_EQ(prey_sensors[11], 2.0f / kMaxDensity);
}

} // namespace
} // namespace moonai
