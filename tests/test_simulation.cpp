#include <gtest/gtest.h>
#include "core/config.hpp"
#include "simulation/simulation_manager.hpp"
#include "simulation/environment.hpp"
#include "simulation/spatial_grid.hpp"
#include "simulation/predator.hpp"
#include "simulation/prey.hpp"
#include "simulation/physics.hpp"

using namespace moonai;

// ── Environment Tests ────────────────────────────────────────────────────

TEST(EnvironmentTest, ClampPosition) {
    SimulationConfig config;
    config.grid_width = 100;
    config.grid_height = 100;
    config.boundary_mode = BoundaryMode::Clamp;
    Environment env(config);

    Vec2 pos = env.apply_boundary({150.0f, -10.0f});
    EXPECT_FLOAT_EQ(pos.x, 100.0f);
    EXPECT_FLOAT_EQ(pos.y, 0.0f);
}

TEST(EnvironmentTest, WrapPosition) {
    SimulationConfig config;
    config.grid_width = 100;
    config.grid_height = 100;
    config.boundary_mode = BoundaryMode::Wrap;
    Environment env(config);

    Vec2 pos = env.apply_boundary({150.0f, -10.0f});
    EXPECT_FLOAT_EQ(pos.x, 50.0f);
    EXPECT_FLOAT_EQ(pos.y, 90.0f);
}

TEST(EnvironmentTest, FoodInitialization) {
    SimulationConfig config;
    config.grid_width = 200;
    config.grid_height = 200;
    Environment env(config);
    Random rng(42);

    env.initialize_food(rng, 50);
    EXPECT_EQ(env.food().size(), 50u);

    for (const auto& f : env.food()) {
        EXPECT_TRUE(f.active);
        EXPECT_GE(f.position.x, 0.0f);
        EXPECT_LE(f.position.x, 200.0f);
    }
}

TEST(EnvironmentTest, EatFood) {
    SimulationConfig config;
    config.grid_width = 100;
    config.grid_height = 100;
    Environment env(config);
    Random rng(42);

    env.initialize_food(rng, 10);
    Vec2 food_pos = env.food()[0].position;

    // Try to eat from exact food position
    bool ate = env.try_eat_food(food_pos, 5.0f);
    EXPECT_TRUE(ate);
    EXPECT_FALSE(env.food()[0].active);

    // Can't eat same food again
    bool ate2 = env.try_eat_food(food_pos, 5.0f);
    EXPECT_FALSE(ate2);
}

// ── Spatial Grid Tests ───────────────────────────────────────────────────

TEST(SpatialGridTest, InsertAndQuery) {
    SpatialGrid grid(100, 100, 25.0f);

    grid.insert(0, {10.0f, 10.0f});
    grid.insert(1, {12.0f, 12.0f});
    grid.insert(2, {90.0f, 90.0f});

    auto near_origin = grid.query({10.0f, 10.0f});
    EXPECT_GE(near_origin.size(), 2u);  // agents 0 and 1

    auto near_corner = grid.query({90.0f, 90.0f});
    bool found_2 = false;
    for (auto id : near_corner) {
        if (id == 2) found_2 = true;
    }
    EXPECT_TRUE(found_2);
}

TEST(SpatialGridTest, QueryRadius) {
    SpatialGrid grid(1000, 1000, 50.0f);

    grid.insert(0, {100.0f, 100.0f});
    grid.insert(1, {110.0f, 100.0f});  // 10 units away
    grid.insert(2, {500.0f, 500.0f});  // far away

    auto results = grid.query_radius({100.0f, 100.0f}, 50.0f);
    // Should include 0 and 1 but not necessarily 2
    bool found_0 = false, found_1 = false;
    for (auto id : results) {
        if (id == 0) found_0 = true;
        if (id == 1) found_1 = true;
    }
    EXPECT_TRUE(found_0);
    EXPECT_TRUE(found_1);
}

TEST(SpatialGridTest, ClearWorks) {
    SpatialGrid grid(100, 100, 25.0f);
    grid.insert(0, {50.0f, 50.0f});

    auto before = grid.query({50.0f, 50.0f});
    EXPECT_FALSE(before.empty());

    grid.clear();
    auto after = grid.query({50.0f, 50.0f});
    EXPECT_TRUE(after.empty());
}

// ── Agent Tests ──────────────────────────────────────────────────────────

TEST(AgentTest, ApplyMovement) {
    Prey prey(0, {50.0f, 50.0f}, 5.0f, 100.0f, 100.0f);
    prey.apply_movement({1.0f, 0.0f}, 1.0f);

    EXPECT_NEAR(prey.position().x, 55.0f, 0.01f);
    EXPECT_NEAR(prey.position().y, 50.0f, 0.01f);
}

TEST(AgentTest, EnergyDrainAndDeath) {
    Prey prey(0, {50.0f, 50.0f}, 5.0f, 100.0f, 10.0f);
    EXPECT_TRUE(prey.alive());
    EXPECT_FALSE(prey.is_dead());

    prey.drain_energy(11.0f);
    EXPECT_TRUE(prey.is_dead());
    EXPECT_LT(prey.energy(), 0.0f);
}

TEST(AgentTest, AgeIncrements) {
    Prey prey(0, {0, 0}, 1.0f, 50.0f, 100.0f);
    EXPECT_EQ(prey.age(), 0);
    prey.update(0.016f);
    EXPECT_EQ(prey.age(), 1);
    prey.update(0.016f);
    EXPECT_EQ(prey.age(), 2);
}

// ── SimulationManager Tests ──────────────────────────────────────────────

TEST(SimulationManagerTest, InitializeCreatesAgents) {
    SimulationConfig config;
    config.predator_count = 5;
    config.prey_count = 10;
    config.seed = 42;

    SimulationManager sim(config);
    sim.initialize();

    EXPECT_EQ(sim.agents().size(), 15u);
    EXPECT_EQ(sim.alive_predators(), 5);
    EXPECT_EQ(sim.alive_prey(), 10);
}

TEST(SimulationManagerTest, TickAdvances) {
    SimulationConfig config;
    config.predator_count = 3;
    config.prey_count = 5;
    config.seed = 42;

    SimulationManager sim(config);
    sim.initialize();

    EXPECT_EQ(sim.current_tick(), 0);
    sim.tick(0.016f);
    EXPECT_EQ(sim.current_tick(), 1);
}

TEST(SimulationManagerTest, SensorsHaveCorrectSize) {
    SimulationConfig config;
    config.predator_count = 5;
    config.prey_count = 10;
    config.seed = 42;

    SimulationManager sim(config);
    sim.initialize();

    auto sensors = sim.get_sensors(0);
    auto vec = sensors.to_vector();
    EXPECT_EQ(static_cast<int>(vec.size()), SensorInput::SIZE);
}

TEST(SimulationManagerTest, ResetRestoresState) {
    SimulationConfig config;
    config.predator_count = 5;
    config.prey_count = 10;
    config.seed = 42;

    SimulationManager sim(config);
    sim.initialize();
    sim.tick(0.016f);
    sim.tick(0.016f);

    EXPECT_EQ(sim.current_tick(), 2);

    sim.reset();
    EXPECT_EQ(sim.current_tick(), 0);
    EXPECT_EQ(sim.agents().size(), 15u);
}

// ── Regression Tests for Bug Fixes ──────────────────────────────────────

TEST(EnvironmentTest, WrapPositionWithLargeNegativeValues) {
    // Regression for Fix 2: fmod(pos + w, w) fails when pos < -w.
    // Old code: fmod(-250 + 100, 100) = fmod(-150, 100) = -50 (bug: still negative)
    // New code: fmod(-250, 100) = -50, then -50 + 100 = 50 (correct)
    SimulationConfig config;
    config.grid_width = 100;
    config.grid_height = 100;
    config.boundary_mode = BoundaryMode::Wrap;
    Environment env(config);

    Vec2 pos = env.apply_boundary({-250.0f, -350.0f});
    EXPECT_FLOAT_EQ(pos.x, 50.0f);
    EXPECT_FLOAT_EQ(pos.y, 50.0f);
}

TEST(PhysicsTest, SensorDistIsNegativeOneWhenNoNeighbors) {
    // Regression for Fix 5: default sentinel should be -1 when no object is in range.
    SimulationConfig config;
    config.grid_width = 1000;
    config.grid_height = 1000;
    config.boundary_mode = BoundaryMode::Clamp;

    SpatialGrid grid(1000, 1000, 50.0f);

    std::vector<std::unique_ptr<Agent>> agents;
    agents.push_back(std::make_unique<Prey>(0, Vec2{500.0f, 500.0f}, 3.5f, 100.0f, 100.0f));
    grid.insert(0, {500.0f, 500.0f});

    std::vector<Food> no_food;
    SpatialGrid food_grid(1000, 1000, 50.0f);

    SensorInput s = Physics::build_sensors(
        *agents[0], agents, no_food, grid, food_grid,
        1000.0f, 1000.0f, 100.0f, true);

    // No predators, no prey, no food in range → dist sentinel should be -1.0f
    EXPECT_FLOAT_EQ(s.nearest_predator_dist, -1.0f);
    EXPECT_FLOAT_EQ(s.nearest_prey_dist, -1.0f);
    EXPECT_FLOAT_EQ(s.nearest_food_dist, -1.0f);
}

// ── Species ID Tests ────────────────────────────────────────────────────

TEST(AgentTest, SpeciesIdDefaultAndSet) {
    Prey prey(0, {0, 0}, 1.0f, 50.0f, 100.0f);
    EXPECT_EQ(prey.species_id(), -1);

    prey.set_species_id(5);
    EXPECT_EQ(prey.species_id(), 5);
}

TEST(SimulationManagerTest, EventsRecordedOnTick) {
    SimulationConfig config;
    config.predator_count = 3;
    config.prey_count = 5;
    config.seed = 42;

    SimulationManager sim(config);
    sim.initialize();

    // Events should be empty or populated after a tick
    sim.tick(1.0f / 60.0f);
    // Just verify the accessor works (events may or may not occur)
    const auto& events = sim.last_events();
    for (const auto& e : events) {
        EXPECT_TRUE(e.type == SimEvent::Kill || e.type == SimEvent::Food);
    }
}
