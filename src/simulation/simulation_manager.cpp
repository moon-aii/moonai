#include "simulation/simulation_manager.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/evolution_manager.hpp"
#include "gpu/ecs_gpu_packing.hpp"
#include "gpu/gpu_batch_ecs.hpp"
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
            std::max(config.vision_range, 1.0f)),
      sensor_system_(grid_, static_cast<float>(config.grid_size),
                     static_cast<float>(config.grid_size),
                     config.initial_energy,
                     config.boundary_mode == BoundaryMode::Clamp),
      energy_system_(config.energy_drain_per_step, config.energy_drain_per_step,
                     static_cast<float>(config.max_steps),
                     config.initial_energy),
      movement_system_(static_cast<float>(config.grid_size),
                       static_cast<float>(config.grid_size),
                       config.boundary_mode == BoundaryMode::Clamp),
      combat_system_(grid_, config.attack_range),
      food_respawn_system_(static_cast<float>(config.grid_size),
                           static_cast<float>(config.grid_size),
                           config.food_respawn_rate, config.seed) {}

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

void SimulationManager::step_ecs(Registry &registry) {
  MOONAI_PROFILE_SCOPE("simulation_step");
  last_events_.clear();

  // Update spatial grid
  rebuild_spatial_grid_ecs(registry);

  // Update sensor inputs for all agents (must happen before
  // compute_actions_ecs)
  sensor_system_.update(registry);

  // Process interactions using EnergySystem
  energy_system_.update(registry);
  process_food_ecs(registry);

  // Process combat using CombatSystem
  combat_system_.update(registry);

  // Convert kill events to SimEvents
  for (const auto &kill : combat_system_.kill_events()) {
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
  combat_system_.clear_events();

  // Respawn food using FoodRespawnSystem
  food_respawn_system_.set_step(current_step_);
  food_respawn_system_.update(registry);

  // Apply movement and boundary conditions using MovementSystem
  {
    MOONAI_PROFILE_SCOPE("boundary_apply");
    movement_system_.update(registry);
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
  MOONAI_PROFILE_SCOPE("rebuild_spatial_grid");
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
  MOONAI_PROFILE_SCOPE("process_food");

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
  MOONAI_PROFILE_SCOPE("death_check");

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
  MOONAI_PROFILE_SCOPE("count_alive");

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

SimulationManager::~SimulationManager() = default;

void SimulationManager::enable_gpu(bool enable) {
  gpu_enabled_ = enable;
  if (enable && !gpu_batch_) {
    // Estimate max entities (predators + prey + food) * 6 for safety margin
    size_t max_entities = static_cast<size_t>(
        (config_.predator_count + config_.prey_count + config_.food_count) * 6);
    gpu_batch_ = std::make_unique<gpu::GpuBatchECS>(max_entities);
    spdlog::info("GPU batch processing enabled with capacity {}", max_entities);
  } else if (!enable) {
    gpu_batch_.reset();
    spdlog::info("GPU batch processing disabled");
  }
}

void SimulationManager::step_gpu_ecs(Registry &registry,
                                     EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_gpu");
  last_events_.clear();

  if (!gpu_batch_ || !gpu_batch_->ok()) {
    spdlog::error(
        "GPU batch not initialized or in error state, falling back to CPU");
    step_ecs(registry);
    return;
  }

  // 1. Prepare GPU buffers - pack ECS data
  size_t count = 0;
  try {
    count = gpu::prepare_ecs_for_gpu(registry, gpu_batch_->mapping(),
                                     gpu_batch_->buffer());
  } catch (const std::exception &ex) {
    spdlog::error("GPU preparation failed: {}. Falling back to CPU step.",
                  ex.what());
    gpu_enabled_ = false;
    step_ecs(registry);
    return;
  }
  if (count == 0) {
    return;
  }

  // 2. Upload to GPU
  gpu_batch_->upload_async(count);

  // 3. Build sensors + inference + vitals/combat on current positions
  gpu::GpuStepParams params;
  params.world_width = static_cast<float>(config_.grid_size);
  params.world_height = static_cast<float>(config_.grid_size);
  params.has_walls = (config_.boundary_mode == BoundaryMode::Clamp);
  params.energy_drain_per_step = config_.energy_drain_per_step;
  params.vision_range = config_.vision_range;
  params.max_energy = static_cast<float>(config_.initial_energy);
  params.max_age = config_.max_steps;
  params.food_pickup_range = config_.food_pickup_range;
  params.attack_range = config_.attack_range;
  params.energy_gain_from_food =
      static_cast<float>(config_.energy_gain_from_food);
  params.energy_gain_from_kill =
      static_cast<float>(config_.energy_gain_from_kill);
  params.food_respawn_rate = config_.food_respawn_rate;
  params.seed = rng_.seed();
  params.step_index = current_step_;

  gpu_batch_->launch_build_sensors_async(params, count);
  evolution.launch_gpu_neural(*gpu_batch_, count);
  gpu_batch_->launch_update_vitals_async(params, count);
  gpu_batch_->launch_process_combat_async(params, count);

  // 4. Download intermediate results from GPU
  gpu_batch_->download_async(count);
  gpu_batch_->synchronize();
  if (!gpu_batch_->ok()) {
    spdlog::error("GPU step failed, disabling GPU path and retrying on CPU");
    gpu_enabled_ = false;
    gpu_batch_.reset();
    step_ecs(registry);
    return;
  }

  // 5. Unpack results back to ECS
  gpu::apply_gpu_results(gpu_batch_->buffer(), gpu_batch_->mapping(), registry);

  auto &stats = registry.stats();
  const auto &positions = registry.positions();
  for (uint32_t gpu_idx = 0; gpu_idx < count; ++gpu_idx) {
    Entity entity = gpu_batch_->mapping().entity_at(gpu_idx);
    if (entity == INVALID_ENTITY) {
      continue;
    }

    const size_t ecs_idx = registry.index_of(entity);
    if (ecs_idx == std::numeric_limits<size_t>::max()) {
      continue;
    }

    const uint32_t kills = gpu_batch_->buffer().host_kill_counts()[gpu_idx];
    if (kills > 0) {
      stats.kills[ecs_idx] += static_cast<int>(kills);
    }

    const int killer_gpu_idx = gpu_batch_->buffer().host_killed_by()[gpu_idx];
    if (killer_gpu_idx >= 0) {
      Entity killer = gpu_batch_->mapping().entity_at(
          static_cast<uint32_t>(killer_gpu_idx));
      Vec2 pos{positions.x[ecs_idx], positions.y[ecs_idx]};
      last_events_.push_back(SimEvent{SimEvent::Kill, killer, entity,
                                      INVALID_ENTITY, INVALID_ENTITY, pos});
    }
  }

  rebuild_spatial_grid_ecs(registry);

  // 6. CPU-only systems that run before movement
  process_food_ecs(registry);
  food_respawn_system_.set_step(current_step_);
  food_respawn_system_.update(registry);

  // 7. Re-upload updated CPU state and run movement last, matching CPU order.
  try {
    count = gpu::prepare_ecs_for_gpu(registry, gpu_batch_->mapping(),
                                     gpu_batch_->buffer());
  } catch (const std::exception &ex) {
    spdlog::error(
        "GPU movement preparation failed: {}. Falling back to CPU step.",
        ex.what());
    gpu_enabled_ = false;
    step_ecs(registry);
    return;
  }
  gpu_batch_->upload_async(count);
  gpu_batch_->launch_apply_movement_async(params, count);
  gpu_batch_->download_async(count);
  gpu_batch_->synchronize();
  if (!gpu_batch_->ok()) {
    spdlog::error(
        "GPU movement step failed, disabling GPU path and retrying on CPU");
    gpu_enabled_ = false;
    gpu_batch_.reset();
    step_ecs(registry);
    return;
  }
  gpu::apply_gpu_results(gpu_batch_->buffer(), gpu_batch_->mapping(), registry);

  process_step_deaths_ecs(registry);

  // Update alive counters
  rebuild_spatial_grid_ecs(registry);
  count_alive_ecs(registry);

  ++current_step_;
}

} // namespace moonai
