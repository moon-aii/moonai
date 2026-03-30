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

Vec2 wrap_diff(Vec2 diff, float world_size) {
  if (std::abs(diff.x) > world_size * 0.5f) {
    diff.x = diff.x > 0.0f ? diff.x - world_size : diff.x + world_size;
  }
  if (std::abs(diff.y) > world_size * 0.5f) {
    diff.y = diff.y > 0.0f ? diff.y - world_size : diff.y + world_size;
  }
  return diff;
}

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
  evolution.initialize(state, AgentSoA::INPUT_COUNT, AgentSoA::OUTPUT_COUNT);
  evolution.seed_initial_population(state);

  ASSERT_EQ(state.predators.size(), 1u);
  ASSERT_EQ(state.prey.size(), 1u);
  ASSERT_EQ(AgentSoA::INPUT_COUNT, 12);

  const std::size_t predator_idx = 0;
  const std::size_t prey_idx = 0;
  state.predators.positions.x[predator_idx] = 10.0f;
  state.predators.positions.y[predator_idx] = 10.0f;
  state.prey.positions.x[prey_idx] = 22.0f;
  state.prey.positions.y[prey_idx] = 16.0f;

  simulation.step(state, evolution);

  const auto &food_store = state.food_store;
  ASSERT_EQ(food_store.size(), 2u);
  ASSERT_TRUE(food_store.active[0]);
  ASSERT_TRUE(food_store.active[1]);

  auto nearest_food_delta = [&](const Vec2 &origin) {
    Vec2 best{0.0f, 0.0f};
    float best_dist = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < food_store.size(); ++i) {
      Vec2 diff = wrap_diff({food_store.positions.x[i] - origin.x,
                             food_store.positions.y[i] - origin.y},
                            static_cast<float>(config.grid_size));
      const float dist = distance_sq(diff);
      if (dist < best_dist) {
        best_dist = dist;
        best = diff;
      }
    }
    return best;
  };

  const Vec2 predator_pos{state.predators.positions.x[predator_idx],
                          state.predators.positions.y[predator_idx]};
  const Vec2 prey_pos{state.prey.positions.x[prey_idx],
                      state.prey.positions.y[prey_idx]};
  const Vec2 predator_to_prey =
      wrap_diff({prey_pos.x - predator_pos.x, prey_pos.y - predator_pos.y},
                static_cast<float>(config.grid_size));
  const Vec2 prey_to_predator =
      wrap_diff({predator_pos.x - prey_pos.x, predator_pos.y - prey_pos.y},
                static_cast<float>(config.grid_size));
  const Vec2 predator_to_food = nearest_food_delta(predator_pos);
  const Vec2 prey_to_food = nearest_food_delta(prey_pos);

  const float *predator_sensors =
      state.predators.agents.input_ptr(predator_idx);
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
  EXPECT_FLOAT_EQ(predator_sensors[11], 2.0f / kMaxDensity);

  const float *prey_sensors = state.prey.agents.input_ptr(prey_idx);
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
  EXPECT_FLOAT_EQ(prey_sensors[11], 2.0f / kMaxDensity);
}

} // namespace
} // namespace moonai
