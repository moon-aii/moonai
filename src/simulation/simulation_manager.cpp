#include "simulation/simulation_manager.hpp"

#include "core/profiler_macros.hpp"
#include "evolution/evolution_manager.hpp"
#include "gpu/ecs_gpu_packing.hpp"
#include "gpu/gpu_batch_ecs.hpp"
#include "simulation/registry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace moonai {

namespace {
constexpr float kPi = 3.14159265f;
constexpr float kMaxDensity = 10.0f;

Vec2 wrap_diff(Vec2 diff, float world_width, float world_height) {
  if (std::abs(diff.x) > world_width * 0.5f) {
    diff.x = diff.x > 0.0f ? diff.x - world_width : diff.x + world_width;
  }
  if (std::abs(diff.y) > world_height * 0.5f) {
    diff.y = diff.y > 0.0f ? diff.y - world_height : diff.y + world_height;
  }
  return diff;
}

float normalize_angle(float dx, float dy) {
  return std::atan2(dy, dx) / kPi;
}

void build_sensors(PackedStepState &state, const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);
  const float vision = config.vision_range;
  const float vision_sq = vision * vision;

  for (std::size_t i = 0; i < state.agents.size(); ++i) {
    float *sensor_ptr = state.agents.sensor_ptr(i);
    sensor_ptr[0] = -1.0f;
    sensor_ptr[1] = 0.0f;
    sensor_ptr[2] = -1.0f;
    sensor_ptr[3] = 0.0f;
    sensor_ptr[4] = -1.0f;
    sensor_ptr[5] = 0.0f;
    sensor_ptr[6] = 1.0f;
    sensor_ptr[7] = 0.0f;
    sensor_ptr[8] = 0.0f;
    sensor_ptr[9] = 0.0f;
    sensor_ptr[10] = 0.0f;
    sensor_ptr[11] = 1.0f;
    sensor_ptr[12] = 1.0f;
    sensor_ptr[13] = 1.0f;
    sensor_ptr[14] = 1.0f;

    if (!state.agents.alive[i]) {
      continue;
    }

    const Vec2 pos{state.agents.pos_x[i], state.agents.pos_y[i]};
    float nearest_pred_dist_sq = std::numeric_limits<float>::max();
    float nearest_prey_dist_sq = std::numeric_limits<float>::max();
    float nearest_food_dist_sq = std::numeric_limits<float>::max();
    Vec2 nearest_pred_dir{0.0f, 0.0f};
    Vec2 nearest_prey_dir{0.0f, 0.0f};
    Vec2 nearest_food_dir{0.0f, 0.0f};
    int local_predators = 0;
    int local_prey = 0;

    for (std::size_t other = 0; other < state.agents.size(); ++other) {
      if (other == i || !state.agents.alive[other]) {
        continue;
      }

      Vec2 diff = wrap_diff({state.agents.pos_x[other] - pos.x,
                             state.agents.pos_y[other] - pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq > vision_sq) {
        continue;
      }

      if (state.agents.type[other] == IdentitySoA::TYPE_PREDATOR) {
        ++local_predators;
        if (dist_sq < nearest_pred_dist_sq) {
          nearest_pred_dist_sq = dist_sq;
          nearest_pred_dir = diff;
        }
      } else {
        ++local_prey;
        if (dist_sq < nearest_prey_dist_sq) {
          nearest_prey_dist_sq = dist_sq;
          nearest_prey_dir = diff;
        }
      }
    }

    if (state.agents.type[i] == IdentitySoA::TYPE_PREY) {
      for (std::size_t food_idx = 0; food_idx < state.foods.size();
           ++food_idx) {
        if (!state.foods.active[food_idx]) {
          continue;
        }

        Vec2 diff = wrap_diff({state.foods.pos_x[food_idx] - pos.x,
                               state.foods.pos_y[food_idx] - pos.y},
                              world_size, world_size);
        const float dist_sq = diff.x * diff.x + diff.y * diff.y;
        if (dist_sq <= vision_sq && dist_sq < nearest_food_dist_sq) {
          nearest_food_dist_sq = dist_sq;
          nearest_food_dir = diff;
        }
      }
    }

    if (nearest_pred_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[0] = std::sqrt(nearest_pred_dist_sq) / vision;
      sensor_ptr[1] = normalize_angle(nearest_pred_dir.x, nearest_pred_dir.y);
    }
    if (nearest_prey_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[2] = std::sqrt(nearest_prey_dist_sq) / vision;
      sensor_ptr[3] = normalize_angle(nearest_prey_dir.x, nearest_prey_dir.y);
    }
    if (nearest_food_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[4] = std::sqrt(nearest_food_dist_sq) / vision;
      sensor_ptr[5] = normalize_angle(nearest_food_dir.x, nearest_food_dir.y);
    }

    sensor_ptr[6] =
        std::clamp(state.agents.energy[i] /
                       (static_cast<float>(config.initial_energy) * 2.0f),
                   0.0f, 1.0f);
    if (state.agents.speed[i] > 0.0f) {
      sensor_ptr[7] = std::clamp(state.agents.vel_x[i] / state.agents.speed[i],
                                 -1.0f, 1.0f);
      sensor_ptr[8] = std::clamp(state.agents.vel_y[i] / state.agents.speed[i],
                                 -1.0f, 1.0f);
    }
    sensor_ptr[9] = std::clamp(
        static_cast<float>(local_predators) / kMaxDensity, 0.0f, 1.0f);
    sensor_ptr[10] =
        std::clamp(static_cast<float>(local_prey) / kMaxDensity, 0.0f, 1.0f);
  }
}

void update_vitals(PackedStepState &state, const SimulationConfig &config) {
  for (std::size_t i = 0; i < state.agents.size(); ++i) {
    if (!state.agents.alive[i]) {
      continue;
    }

    state.agents.age[i] += 1;
    if (state.agents.reproduction_cooldown[i] > 0) {
      state.agents.reproduction_cooldown[i] -= 1;
    }

    state.agents.energy[i] -= config.energy_drain_per_step;
    const bool died_of_starvation = state.agents.energy[i] <= 0.0f;
    const bool died_of_age =
        config.max_steps > 0 && state.agents.age[i] >= config.max_steps;
    if (died_of_starvation || died_of_age) {
      state.agents.energy[i] = 0.0f;
      state.agents.alive[i] = 0;
    }
  }
}

void process_food(PackedStepState &state, const SimulationConfig &config) {
  std::fill(state.foods.consumed_by.begin(), state.foods.consumed_by.end(), -1);
  const float world_size = static_cast<float>(config.grid_size);
  const float range_sq = config.food_pickup_range * config.food_pickup_range;

  for (std::size_t prey_idx = 0; prey_idx < state.agents.size(); ++prey_idx) {
    if (!state.agents.alive[prey_idx] ||
        state.agents.type[prey_idx] != IdentitySoA::TYPE_PREY) {
      continue;
    }

    int best_food = -1;
    float best_dist_sq = range_sq;
    const Vec2 prey_pos{state.agents.pos_x[prey_idx],
                        state.agents.pos_y[prey_idx]};

    for (std::size_t food_idx = 0; food_idx < state.foods.size(); ++food_idx) {
      if (!state.foods.active[food_idx]) {
        continue;
      }

      Vec2 diff = wrap_diff({state.foods.pos_x[food_idx] - prey_pos.x,
                             state.foods.pos_y[food_idx] - prey_pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq <= best_dist_sq) {
        best_dist_sq = dist_sq;
        best_food = static_cast<int>(food_idx);
      }
    }

    if (best_food >= 0) {
      int &owner = state.foods.consumed_by[static_cast<std::size_t>(best_food)];
      if (owner < 0 || static_cast<int>(prey_idx) < owner) {
        owner = static_cast<int>(prey_idx);
      }
    }
  }

  for (std::size_t food_idx = 0; food_idx < state.foods.size(); ++food_idx) {
    const int prey_idx = state.foods.consumed_by[food_idx];
    if (!state.foods.active[food_idx] || prey_idx < 0 ||
        !state.agents.alive[static_cast<std::size_t>(prey_idx)]) {
      continue;
    }

    state.foods.active[food_idx] = 0;
    state.agents.energy[static_cast<std::size_t>(prey_idx)] +=
        config.energy_gain_from_food;
  }
}

void process_combat(PackedStepState &state, const SimulationConfig &config) {
  std::fill(state.agents.killed_by.begin(), state.agents.killed_by.end(), -1);
  std::fill(state.agents.kill_counts.begin(), state.agents.kill_counts.end(),
            0U);

  const float world_size = static_cast<float>(config.grid_size);
  const float range_sq = config.attack_range * config.attack_range;

  for (std::size_t predator_idx = 0; predator_idx < state.agents.size();
       ++predator_idx) {
    if (!state.agents.alive[predator_idx] ||
        state.agents.type[predator_idx] != IdentitySoA::TYPE_PREDATOR) {
      continue;
    }

    int best_prey = -1;
    float best_dist_sq = range_sq;
    const Vec2 predator_pos{state.agents.pos_x[predator_idx],
                            state.agents.pos_y[predator_idx]};

    for (std::size_t prey_idx = 0; prey_idx < state.agents.size(); ++prey_idx) {
      if (!state.agents.alive[prey_idx] ||
          state.agents.type[prey_idx] != IdentitySoA::TYPE_PREY) {
        continue;
      }

      Vec2 diff = wrap_diff({state.agents.pos_x[prey_idx] - predator_pos.x,
                             state.agents.pos_y[prey_idx] - predator_pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq <= best_dist_sq) {
        best_dist_sq = dist_sq;
        best_prey = static_cast<int>(prey_idx);
      }
    }

    if (best_prey >= 0) {
      int &killer = state.agents.killed_by[static_cast<std::size_t>(best_prey)];
      if (killer < 0 || static_cast<int>(predator_idx) < killer) {
        killer = static_cast<int>(predator_idx);
      }
    }
  }

  for (std::size_t prey_idx = 0; prey_idx < state.agents.size(); ++prey_idx) {
    const int killer_idx = state.agents.killed_by[prey_idx];
    if (!state.agents.alive[prey_idx] ||
        state.agents.type[prey_idx] != IdentitySoA::TYPE_PREY ||
        killer_idx < 0 ||
        !state.agents.alive[static_cast<std::size_t>(killer_idx)]) {
      continue;
    }

    state.agents.alive[prey_idx] = 0;
    state.agents.energy[static_cast<std::size_t>(killer_idx)] +=
        config.energy_gain_from_kill;
    state.agents.kill_counts[static_cast<std::size_t>(killer_idx)] += 1;
  }
}

void apply_movement(PackedStepState &state, const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);

  for (std::size_t i = 0; i < state.agents.size(); ++i) {
    if (!state.agents.alive[i]) {
      continue;
    }

    float dx = state.agents.brain_ptr(i)[0];
    float dy = state.agents.brain_ptr(i)[1];
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len > 1e-6f) {
      dx /= len;
      dy /= len;
    } else {
      dx = 0.0f;
      dy = 0.0f;
    }

    state.agents.vel_x[i] = dx * state.agents.speed[i];
    state.agents.vel_y[i] = dy * state.agents.speed[i];

    const float old_x = state.agents.pos_x[i];
    const float old_y = state.agents.pos_y[i];

    state.agents.pos_x[i] += state.agents.vel_x[i];
    state.agents.pos_y[i] += state.agents.vel_y[i];

    while (state.agents.pos_x[i] < 0.0f)
      state.agents.pos_x[i] += world_size;
    while (state.agents.pos_x[i] >= world_size)
      state.agents.pos_x[i] -= world_size;
    while (state.agents.pos_y[i] < 0.0f)
      state.agents.pos_y[i] += world_size;
    while (state.agents.pos_y[i] >= world_size)
      state.agents.pos_y[i] -= world_size;

    Vec2 move = wrap_diff(
        {state.agents.pos_x[i] - old_x, state.agents.pos_y[i] - old_y},
        world_size, world_size);
    state.agents.distance_traveled[i] +=
        std::sqrt(move.x * move.x + move.y * move.y);
  }
}
} // namespace

SimulationManager::SimulationManager(const SimulationConfig &config)
    : config_(config),
      rng_(config.seed != 0
               ? config.seed
               : static_cast<std::uint64_t>(std::chrono::steady_clock::now()
                                                .time_since_epoch()
                                                .count())),
      grid_(config.grid_size, config.grid_size,
            std::max(config.mate_range, 1.0f)),
      food_grid_(config.grid_size, config.grid_size,
                 std::max(config.vision_range, 1.0f)) {}

void SimulationManager::initialize() {
  initialize(true);
}

void SimulationManager::initialize(bool log_initialization) {
  current_step_ = 0;
  food_store_.initialize(config_, rng_);
  rebuild_food_grid();

  if (log_initialization) {
    spdlog::info("Simulation initialized: {} food pellets (seed: {})",
                 config_.food_count, rng_.seed());
  }
}

PackedStepState
SimulationManager::pack_step_state(const Registry &registry) const {
  PackedStepState state;
  const auto &living = registry.living_entities();
  state.agents.resize(living.size());
  state.foods.resize(food_store_.size());

  const auto &positions = registry.positions();
  const auto &motion = registry.motion();
  const auto &vitals = registry.vitals();
  const auto &identity = registry.identity();
  const auto &stats = registry.stats();
  const auto &sensors = registry.sensors();
  const auto &brain = registry.brain();

  for (std::size_t agent_idx = 0; agent_idx < living.size(); ++agent_idx) {
    const Entity entity = living[agent_idx];
    const std::size_t idx = registry.index_of(entity);

    state.agents.entities[agent_idx] = entity;
    state.agents.pos_x[agent_idx] = positions.x[idx];
    state.agents.pos_y[agent_idx] = positions.y[idx];
    state.agents.vel_x[agent_idx] = motion.vel_x[idx];
    state.agents.vel_y[agent_idx] = motion.vel_y[idx];
    state.agents.speed[agent_idx] = motion.speed[idx];
    state.agents.energy[agent_idx] = vitals.energy[idx];
    state.agents.age[agent_idx] = vitals.age[idx];
    state.agents.alive[agent_idx] = vitals.alive[idx];
    state.agents.was_alive[agent_idx] = vitals.alive[idx];
    state.agents.type[agent_idx] = identity.type[idx];
    state.agents.reproduction_cooldown[agent_idx] =
        vitals.reproduction_cooldown[idx];
    state.agents.distance_traveled[agent_idx] = stats.distance_traveled[idx];
    state.agents.kill_counts[agent_idx] = 0;
    state.agents.killed_by[agent_idx] = -1;
    std::copy_n(sensors.input_ptr(idx), SensorSoA::INPUT_COUNT,
                state.agents.sensor_ptr(agent_idx));
    state.agents.brain_ptr(agent_idx)[0] = brain.decision_x[idx];
    state.agents.brain_ptr(agent_idx)[1] = brain.decision_y[idx];
  }

  for (std::size_t food_idx = 0; food_idx < food_store_.size(); ++food_idx) {
    state.foods.pos_x[food_idx] = food_store_.pos_x()[food_idx];
    state.foods.pos_y[food_idx] = food_store_.pos_y()[food_idx];
    state.foods.active[food_idx] = food_store_.active()[food_idx];
    state.foods.was_active[food_idx] = food_store_.active()[food_idx];
    state.foods.slot_index[food_idx] = food_store_.slot_index()[food_idx];
    state.foods.consumed_by[food_idx] = -1;
  }

  return state;
}

void SimulationManager::apply_step_state(Registry &registry,
                                         const PackedStepState &state) {
  auto &positions = registry.positions();
  auto &motion = registry.motion();
  auto &vitals = registry.vitals();
  auto &stats = registry.stats();
  auto &sensors = registry.sensors();
  auto &brain = registry.brain();

  for (std::size_t i = 0; i < state.agents.size(); ++i) {
    const std::size_t idx = registry.index_of(state.agents.entities[i]);
    positions.x[idx] = state.agents.pos_x[i];
    positions.y[idx] = state.agents.pos_y[i];
    motion.vel_x[idx] = state.agents.vel_x[i];
    motion.vel_y[idx] = state.agents.vel_y[i];
    vitals.energy[idx] = state.agents.energy[i];
    vitals.age[idx] = state.agents.age[i];
    vitals.alive[idx] = state.agents.alive[i];
    vitals.reproduction_cooldown[idx] = state.agents.reproduction_cooldown[i];
    stats.distance_traveled[idx] = state.agents.distance_traveled[i];
    std::copy_n(state.agents.sensor_ptr(i), SensorSoA::INPUT_COUNT,
                sensors.input_ptr(idx));
    brain.decision_x[idx] = state.agents.brain_ptr(i)[0];
    brain.decision_y[idx] = state.agents.brain_ptr(i)[1];
  }

  for (std::size_t i = 0; i < state.foods.size(); ++i) {
    food_store_.pos_x()[i] = state.foods.pos_x[i];
    food_store_.pos_y()[i] = state.foods.pos_y[i];
    food_store_.active()[i] = state.foods.active[i];
  }
}

void SimulationManager::run_cpu_backend(PackedStepState &state,
                                        EvolutionManager &evolution) {
  state.clear_transients();
  build_sensors(state, config_);
  evolution.compute_actions_batch(state.agents.entities,
                                  state.agents.sensor_inputs,
                                  state.agents.brain_outputs);
  update_vitals(state, config_);
  process_food(state, config_);
  process_combat(state, config_);
  apply_movement(state, config_);
}

void SimulationManager::collect_step_events(Registry &registry,
                                            const PackedStepState &state,
                                            std::vector<SimEvent> &events) {
  auto &stats = registry.stats();
  const auto &positions = registry.positions();

  for (std::size_t food_idx = 0; food_idx < state.foods.size(); ++food_idx) {
    const int prey_idx = state.foods.consumed_by[food_idx];
    if (!state.foods.was_active[food_idx] || state.foods.active[food_idx] ||
        prey_idx < 0 ||
        static_cast<std::size_t>(prey_idx) >= state.agents.size()) {
      continue;
    }

    const Entity prey =
        state.agents.entities[static_cast<std::size_t>(prey_idx)];
    const std::size_t ecs_idx = registry.index_of(prey);
    stats.food_eaten[ecs_idx] += 1;
    Vec2 pos{positions.x[ecs_idx], positions.y[ecs_idx]};
    events.push_back(SimEvent{SimEvent::Food, prey, INVALID_ENTITY,
                              INVALID_ENTITY, INVALID_ENTITY, pos});
  }

  for (std::size_t agent_idx = 0; agent_idx < state.agents.size();
       ++agent_idx) {
    const Entity entity = state.agents.entities[agent_idx];
    const std::size_t ecs_idx = registry.index_of(entity);

    if (state.agents.kill_counts[agent_idx] > 0) {
      stats.kills[ecs_idx] +=
          static_cast<int>(state.agents.kill_counts[agent_idx]);
    }

    if (state.agents.killed_by[agent_idx] >= 0 &&
        static_cast<std::size_t>(state.agents.killed_by[agent_idx]) <
            state.agents.size()) {
      const Entity killer = state.agents.entities[static_cast<std::size_t>(
          state.agents.killed_by[agent_idx])];
      Vec2 pos{positions.x[ecs_idx], positions.y[ecs_idx]};
      events.push_back(SimEvent{SimEvent::Kill, killer, entity, INVALID_ENTITY,
                                INVALID_ENTITY, pos});
      continue;
    }

    if (state.agents.was_alive[agent_idx] && !state.agents.alive[agent_idx]) {
      Vec2 pos{positions.x[ecs_idx], positions.y[ecs_idx]};
      events.push_back(SimEvent{SimEvent::Death, entity, entity, INVALID_ENTITY,
                                INVALID_ENTITY, pos});
    }
  }
}

void SimulationManager::refresh_world_state_after_step(Registry &registry) {
  food_store_.respawn_step(config_, current_step_, rng_.seed());
  rebuild_food_grid();
  rebuild_spatial_grid_ecs(registry);
  count_alive_ecs(registry);
  ++current_step_;
}

void SimulationManager::rebuild_food_grid() {
  food_grid_.clear();
  for (std::size_t i = 0; i < food_store_.size(); ++i) {
    if (!food_store_.active()[i]) {
      continue;
    }
    food_grid_.insert(Entity{static_cast<uint32_t>(i + 1), 1},
                      Vec2{food_store_.pos_x()[i], food_store_.pos_y()[i]});
  }
}

SimulationManager::SimulationStepResult
SimulationManager::step_ecs(Registry &registry, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_cpu");
  SimulationStepResult result;

  PackedStepState state = pack_step_state(registry);
  run_cpu_backend(state, evolution);
  apply_step_state(registry, state);
  collect_step_events(registry, state, result.events);
  refresh_world_state_after_step(registry);
  result.reproduction_pairs = find_reproduction_pairs(registry);
  return result;
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

  for (Entity entity : living) {
    const std::size_t idx = registry.index_of(entity);
    if (!vitals.alive[idx]) {
      continue;
    }
    grid_.insert(entity, Vec2{positions.x[idx], positions.y[idx]});
  }
}

std::vector<SimulationManager::ReproductionPair>
SimulationManager::find_reproduction_pairs(const Registry &registry) const {
  std::vector<ReproductionPair> pairs;
  std::unordered_set<Entity, EntityHash> used;

  const auto &living = registry.living_entities();
  const auto &positions = registry.positions();
  const auto &vitals = registry.vitals();
  const auto &identity = registry.identity();

  for (Entity entity : living) {
    const std::size_t idx = registry.index_of(entity);
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

      const std::size_t mate_idx = registry.index_of(mate_id);
      if (!vitals.alive[mate_idx] ||
          identity.type[mate_idx] != identity.type[idx] ||
          vitals.energy[mate_idx] < config_.reproduction_energy_threshold ||
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
      const std::size_t mate_idx = registry.index_of(best_mate);
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
    const std::size_t idx = registry.index_of(entity);
    if (!vitals.alive[idx]) {
      continue;
    }
    if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
      ++alive_predators_;
    } else {
      ++alive_prey_;
    }
  }
}

void SimulationManager::refresh_state_ecs(Registry &registry) {
  rebuild_spatial_grid_ecs(registry);
  rebuild_food_grid();
  count_alive_ecs(registry);
}

SimulationManager::~SimulationManager() = default;

void SimulationManager::enable_gpu(bool enable) {
  gpu_enabled_ = enable;
  if (enable && !gpu_batch_) {
    const size_t max_agents =
        static_cast<size_t>((config_.predator_count + config_.prey_count) * 6);
    const size_t max_food = static_cast<size_t>(config_.food_count);
    gpu_batch_ = std::make_unique<gpu::GpuBatchECS>(max_agents, max_food);
    spdlog::info(
        "GPU batch processing enabled with capacities {} agents / {} food",
        max_agents, max_food);
  } else if (!enable) {
    gpu_batch_.reset();
    spdlog::info("GPU batch processing disabled");
  }
}

SimulationManager::SimulationStepResult
SimulationManager::step_gpu_ecs(Registry &registry,
                                EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_gpu");
  SimulationStepResult result;

  if (!gpu_batch_ || !gpu_batch_->ok()) {
    spdlog::error(
        "GPU batch not initialized or in error state, falling back to CPU");
    return step_ecs(registry, evolution);
  }

  PackedStepState state = pack_step_state(registry);

  try {
    gpu::prepare_step_state_for_gpu(state, gpu_batch_->agent_mapping(),
                                    gpu_batch_->food_mapping(),
                                    gpu_batch_->buffer());
  } catch (const std::exception &ex) {
    spdlog::error("GPU preparation failed: {}. Falling back to CPU step.",
                  ex.what());
    gpu_enabled_ = false;
    return step_ecs(registry, evolution);
  }

  const std::size_t agent_count = state.agents.size();
  const std::size_t food_count = state.foods.size();
  if (agent_count == 0) {
    return result;
  }

  gpu::GpuStepParams params;
  params.world_width = static_cast<float>(config_.grid_size);
  params.world_height = static_cast<float>(config_.grid_size);
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

  gpu_batch_->upload_async(agent_count, food_count);
  gpu_batch_->launch_build_sensors_async(params, agent_count, food_count);
  evolution.launch_gpu_neural(*gpu_batch_, agent_count);
  gpu_batch_->launch_post_inference_async(params, agent_count, food_count);
  gpu_batch_->download_async(agent_count, food_count);
  gpu_batch_->synchronize();

  if (!gpu_batch_->ok()) {
    spdlog::error("GPU step failed, disabling GPU path and retrying on CPU");
    gpu_enabled_ = false;
    gpu_batch_.reset();
    return step_ecs(registry, evolution);
  }

  gpu::apply_gpu_results(gpu_batch_->buffer(), gpu_batch_->agent_mapping(),
                         gpu_batch_->food_mapping(), state);
  apply_step_state(registry, state);
  collect_step_events(registry, state, result.events);
  refresh_world_state_after_step(registry);
  result.reproduction_pairs = find_reproduction_pairs(registry);
  return result;
}

} // namespace moonai
