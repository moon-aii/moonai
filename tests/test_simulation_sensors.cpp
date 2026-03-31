#include "core/app_state.hpp"
#include "core/random.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/simulation_manager.hpp"
#include "simulation/simulation_step_systems.hpp"

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
  evolution.initialize(state, simulation_detail::SENSOR_COUNT,
                       simulation_detail::OUTPUT_COUNT);
  evolution.seed_initial_population(state);

  ASSERT_EQ(state.predator.size(), 1u);
  ASSERT_EQ(state.prey.size(), 1u);
  ASSERT_EQ(simulation_detail::SENSOR_COUNT, 14);

  const std::size_t predator_idx = 0;
  const std::size_t prey_idx = 0;
  state.predator.pos_x[predator_idx] = 10.0f;
  state.predator.pos_y[predator_idx] = 10.0f;
  state.prey.pos_x[prey_idx] = 22.0f;
  state.prey.pos_y[prey_idx] = 16.0f;

  std::vector<float> predator_sensors;
  std::vector<float> prey_sensors;
  simulation_detail::build_sensors(state.predator, state.predator, state.prey,
                                   state.food, config, config.predator_speed,
                                   predator_sensors);
  simulation_detail::build_sensors(state.prey, state.predator, state.prey,
                                   state.food, config, config.prey_speed,
                                   prey_sensors);

  const auto &food_store = state.food;
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

  const Vec2 predator_pos{state.predator.pos_x[predator_idx],
                          state.predator.pos_y[predator_idx]};
  const Vec2 prey_pos{state.prey.pos_x[prey_idx], state.prey.pos_y[prey_idx]};
  const Vec2 predator_to_prey{prey_pos.x - predator_pos.x,
                              prey_pos.y - predator_pos.y};
  const Vec2 prey_to_predator{predator_pos.x - prey_pos.x,
                              predator_pos.y - prey_pos.y};
  const Vec2 predator_to_food = nearest_food_delta(predator_pos);
  const Vec2 prey_to_food = nearest_food_delta(prey_pos);

  const float *predator_sensor_ptr =
      &predator_sensors[predator_idx * simulation_detail::SENSOR_COUNT];
  EXPECT_FLOAT_EQ(predator_sensor_ptr[0], kMissingTargetSentinel);
  EXPECT_FLOAT_EQ(predator_sensor_ptr[1], kMissingTargetSentinel);
  EXPECT_NEAR(predator_sensor_ptr[2], predator_to_prey.x / config.vision_range,
              kEpsilon);
  EXPECT_NEAR(predator_sensor_ptr[3], predator_to_prey.y / config.vision_range,
              kEpsilon);
  EXPECT_NEAR(predator_sensor_ptr[4], predator_to_food.x / config.vision_range,
              kEpsilon);
  EXPECT_NEAR(predator_sensor_ptr[5], predator_to_food.y / config.vision_range,
              kEpsilon);
  EXPECT_FLOAT_EQ(predator_sensor_ptr[6], 0.5f);
  EXPECT_FLOAT_EQ(predator_sensor_ptr[7], 0.0f);
  EXPECT_FLOAT_EQ(predator_sensor_ptr[8], 0.0f);
  EXPECT_FLOAT_EQ(predator_sensor_ptr[9], 0.0f);
  EXPECT_FLOAT_EQ(predator_sensor_ptr[10], 1.0f / kMaxDensity);
  // With bounded world (no wrapping), food visibility depends on actual
  // distance Prior value assumed wrapping behavior:
  // EXPECT_FLOAT_EQ(predator_sensor_ptr[11], 2.0f / kMaxDensity);
  EXPECT_FLOAT_EQ(predator_sensor_ptr[11], 1.0f / kMaxDensity);

  const float *prey_sensor_ptr =
      &prey_sensors[prey_idx * simulation_detail::SENSOR_COUNT];
  EXPECT_NEAR(prey_sensor_ptr[0], prey_to_predator.x / config.vision_range,
              kEpsilon);
  EXPECT_NEAR(prey_sensor_ptr[1], prey_to_predator.y / config.vision_range,
              kEpsilon);
  EXPECT_FLOAT_EQ(prey_sensor_ptr[2], kMissingTargetSentinel);
  EXPECT_FLOAT_EQ(prey_sensor_ptr[3], kMissingTargetSentinel);
  EXPECT_NEAR(prey_sensor_ptr[4], prey_to_food.x / config.vision_range,
              kEpsilon);
  EXPECT_NEAR(prey_sensor_ptr[5], prey_to_food.y / config.vision_range,
              kEpsilon);
  EXPECT_FLOAT_EQ(prey_sensor_ptr[6], 0.5f);
  EXPECT_FLOAT_EQ(prey_sensor_ptr[7], 0.0f);
  EXPECT_FLOAT_EQ(prey_sensor_ptr[8], 0.0f);
  EXPECT_FLOAT_EQ(prey_sensor_ptr[9], 1.0f / kMaxDensity);
  EXPECT_FLOAT_EQ(prey_sensor_ptr[10], 0.0f);
  // With bounded world (no wrapping), food visibility depends on actual
  // positions Prey at (22, 16) can see both food items within vision_range=100
  EXPECT_FLOAT_EQ(prey_sensor_ptr[11], 2.0f / kMaxDensity);
}

} // namespace
} // namespace moonai
