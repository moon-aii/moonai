#include "simulation/simulation_manager.hpp"
#include "core/profiler_types.hpp"
#include "simulation/registry.hpp"
#include "simulation/spatial_grid_ecs.hpp"
#include "simulation/systems/combat.hpp"
#include "simulation/systems/movement.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <spdlog/spdlog.h>
#include <unordered_set>

#ifdef MOONAI_OPENMP_ENABLED
#include <omp.h>
#endif

namespace moonai {

SimulationManager::SimulationManager(const SimulationConfig &config)
    : config_(config),
      rng_(config.seed != 0
               ? config.seed
               : static_cast<std::uint64_t>(std::chrono::steady_clock::now()
                                                .time_since_epoch()
                                                .count())),
      grid_(config.grid_size, config.grid_size,
            std::max(config.vision_range, 1.0f)) {
  // Initialize ECS systems
  bool has_walls = (config.boundary_mode == BoundaryMode::Clamp);
  sensor_system_ = std::make_unique<SensorSystem>(
      grid_, static_cast<float>(config.grid_size),
      static_cast<float>(config.grid_size), config.initial_energy, has_walls);
  energy_system_ = std::make_unique<EnergySystem>(
      config.energy_drain_per_step, config.energy_drain_per_step,
      static_cast<float>(config.max_steps), config.initial_energy);
  movement_system_ = std::make_unique<MovementSystem>(
      static_cast<float>(config.grid_size),
      static_cast<float>(config.grid_size), has_walls);
  combat_system_ = std::make_unique<CombatSystem>(grid_, config.attack_range);
  food_respawn_system_ = std::make_unique<FoodRespawnSystem>(
      static_cast<float>(config.grid_size),
      static_cast<float>(config.grid_size), config.food_respawn_rate,
      config.seed);
}

void SimulationManager::initialize() {
  initialize(true);
}

void SimulationManager::initialize(bool log_initialization) {
  current_step_ = 0;
  last_events_.clear();

  // Food is now initialized as ECS entities in seed_initial_population_ecs
  // The EvolutionManager creates food entities when seeding the population

  if (log_initialization) {
    spdlog::info("Simulation initialized: {} food pellets (seed: {})",
                 config_.food_count, rng_.seed());
  }
}

void SimulationManager::step_ecs(Registry &registry, float dt) {
  MOONAI_PROFILE_SCOPE(ProfileEvent::SimulationStep);
  last_events_.clear();

  // Update spatial grid
  rebuild_spatial_grid_ecs(registry);

  // Update sensor inputs for all agents (must happen before
  // compute_actions_ecs)
  if (sensor_system_) {
    sensor_system_->update(registry, dt);
  }

  // Process interactions using EnergySystem
  if (energy_system_) {
    energy_system_->update(registry, dt);
  }
  process_food_ecs(registry);

  // Process combat using CombatSystem
  if (combat_system_) {
    combat_system_->update(registry, dt);

    // Convert kill events to SimEvents
    for (const auto &kill : combat_system_->kill_events()) {
      size_t victim_idx = registry.index_of(kill.victim);
      const auto &positions = registry.positions();
      Vec2 pos{positions.x[victim_idx], positions.y[victim_idx]};

      // Reward the predator with energy
      size_t killer_idx = registry.index_of(kill.killer);
      auto &vitals = registry.vitals();
      auto &stats = registry.stats();
      vitals.energy[killer_idx] += config_.energy_gain_from_kill;
      stats.kills[killer_idx]++;

      last_events_.push_back(SimEvent{SimEvent::Kill, kill.killer, kill.victim,
                                      INVALID_ENTITY, INVALID_ENTITY, pos});
    }
    combat_system_->clear_events();
  }

  // Respawn food using FoodRespawnSystem
  if (food_respawn_system_) {
    food_respawn_system_->set_step(current_step_);
    food_respawn_system_->update(registry, dt);
  }

  // Apply movement and boundary conditions using MovementSystem
  if (movement_system_) {
    MOONAI_PROFILE_SCOPE(ProfileEvent::BoundaryApply);
    movement_system_->update(registry, dt);
  }

  process_step_deaths_ecs(registry);

  rebuild_spatial_grid_ecs(registry);
  count_alive_ecs(registry);
  ++current_step_;
}

void SimulationManager::reset() {
  initialize(false);
}

void SimulationManager::rebuild_spatial_grid_ecs(const Registry &registry) {
  MOONAI_PROFILE_SCOPE(ProfileEvent::RebuildSpatialGrid);
  grid_.clear();

  const auto &living = registry.living_entities();
  const auto &positions = registry.positions();
  const auto &vitals = registry.vitals();
  const auto &food_state = registry.food_state();
  const auto &identity = registry.identity();

  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    // Agents: use vitals.alive, Food: use food_state.active
    bool is_alive = vitals.alive[idx];
    bool is_active_food = (identity.type[idx] == IdentitySoA::TYPE_FOOD) &&
                          food_state.active[idx];
    if (is_alive || is_active_food) {
      Vec2 pos{positions.x[idx], positions.y[idx]};
      grid_.insert(entity, pos);
    }
  }
}

void SimulationManager::process_food_ecs(Registry &registry) {
  MOONAI_PROFILE_SCOPE(ProfileEvent::ProcessFood);

  float eat_range = config_.food_pickup_range;
  float eat_range_sq = eat_range * eat_range;
  const auto &living = registry.living_entities();
  const auto &positions = registry.positions();
  const auto &identity = registry.identity();
  auto &vitals = registry.vitals();
  auto &stats = registry.stats();
  auto &food_state = registry.food_state();

  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    if (!vitals.alive[idx]) {
      continue;
    }

    // Only prey can eat food
    if (identity.type[idx] != IdentitySoA::TYPE_PREY) {
      continue;
    }

    Vec2 pos{positions.x[idx], positions.y[idx]};
    Entity eaten_food = INVALID_ENTITY;

    // Query nearby food from spatial grid
    auto nearby = grid_.query_radius(pos, eat_range);
    bool ate_food = false;

    for (Entity other_e : nearby) {
      size_t other_idx = registry.index_of(other_e);
      if (other_idx == std::numeric_limits<size_t>::max()) {
        continue;
      }

      // Check if this is active food
      if (identity.type[other_idx] != IdentitySoA::TYPE_FOOD ||
          !food_state.active[other_idx]) {
        continue;
      }

      // Check distance
      float dx = positions.x[other_idx] - pos.x;
      float dy = positions.y[other_idx] - pos.y;
      float dist_sq = dx * dx + dy * dy;

      if (dist_sq <= eat_range_sq) {
        // Eat the food
        food_state.active[other_idx] =
            0; // Deactivate food (will respawn later)
        eaten_food = other_e;
        ate_food = true;
        break;
      }
    }

    if (ate_food) {
      vitals.energy[idx] += config_.energy_gain_from_food;
      stats.food_eaten[idx]++;
      last_events_.push_back(SimEvent{SimEvent::Food, entity, eaten_food,
                                      INVALID_ENTITY, INVALID_ENTITY, pos});
    }
  }
}

void SimulationManager::process_step_deaths_ecs(Registry &registry) {
  MOONAI_PROFILE_SCOPE(ProfileEvent::DeathCheck);

  const auto &living = registry.living_entities();
  auto &vitals = registry.vitals();
  const auto &positions = registry.positions();

  std::vector<Entity> dead_entities;

  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    if (!vitals.alive[idx]) {
      continue;
    }

    // Check death conditions
    bool died_of_age =
        (config_.max_steps > 0 && vitals.age[idx] >= config_.max_steps);
    if (vitals.energy[idx] <= 0.0f || died_of_age) {
      vitals.alive[idx] = 0;

      Vec2 pos{positions.x[idx], positions.y[idx]};
      last_events_.push_back(SimEvent{SimEvent::Death, entity, entity,
                                      INVALID_ENTITY, INVALID_ENTITY, pos});
      dead_entities.push_back(entity);
    }
  }

  // Grid will be rebuilt next frame in rebuild_spatial_grid_ecs
  (void)dead_entities;
}

std::vector<SimulationManager::ReproductionPair>
SimulationManager::find_reproduction_pairs_ecs(const Registry &registry) const {
  std::vector<ReproductionPair> pairs;
  std::unordered_set<Entity, EntityHash> used;

  const auto &living = registry.living_entities();
  const auto &positions = registry.positions();
  const auto &vitals = registry.vitals();
  const auto &identity = registry.identity();

  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);

    if (!vitals.alive[idx] ||
        vitals.energy[idx] < config_.reproduction_energy_threshold ||
        vitals.age[idx] < config_.min_reproductive_age_steps ||
        vitals.reproduction_cooldown[idx] > 0 ||
        used.find(entity) != used.end()) {
      continue;
    }

    Vec2 pos{positions.x[idx], positions.y[idx]};
    std::vector<Entity> nearby = grid_.query_radius(pos, config_.mate_range);

    Entity best_mate = INVALID_ENTITY;
    float best_dist_sq = config_.mate_range * config_.mate_range;

    for (Entity mate_id : nearby) {
      if (mate_id == entity || used.find(mate_id) != used.end()) {
        continue;
      }

      size_t mate_idx = registry.index_of(mate_id);
      if (!vitals.alive[mate_idx]) {
        continue;
      }

      // Must be same type
      if (identity.type[mate_idx] != identity.type[idx]) {
        continue;
      }

      // Must meet reproduction criteria
      if (vitals.energy[mate_idx] < config_.reproduction_energy_threshold ||
          vitals.age[mate_idx] < config_.min_reproductive_age_steps ||
          vitals.reproduction_cooldown[mate_idx] > 0) {
        continue;
      }

      Vec2 mate_pos{positions.x[mate_idx], positions.y[mate_idx]};
      Vec2 diff = mate_pos - pos;
      float dist_sq = diff.x * diff.x + diff.y * diff.y;

      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_mate = mate_id;
      }
    }

    if (best_mate != INVALID_ENTITY) {
      size_t mate_idx = registry.index_of(best_mate);
      Vec2 mate_pos{positions.x[mate_idx], positions.y[mate_idx]};
      Vec2 spawn_pos{(pos.x + mate_pos.x) * 0.5f, (pos.y + mate_pos.y) * 0.5f};

      pairs.push_back({entity, best_mate, spawn_pos});
      used.insert(entity);
      used.insert(best_mate);
    }
  }

  return pairs;
}

void SimulationManager::count_alive_ecs(const Registry &registry) {
  MOONAI_PROFILE_SCOPE(ProfileEvent::CountAlive);

  alive_predators_ = 0;
  alive_prey_ = 0;

  const auto &living = registry.living_entities();
  const auto &vitals = registry.vitals();
  const auto &identity = registry.identity();

  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    if (vitals.alive[idx]) {
      if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
        alive_predators_++;
      } else {
        alive_prey_++;
      }
    }
  }
}

void SimulationManager::refresh_state_ecs(Registry &registry) {
  rebuild_spatial_grid_ecs(registry);
  count_alive_ecs(registry);
}

} // namespace moonai
