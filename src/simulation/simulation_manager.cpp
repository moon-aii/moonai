#include "simulation/simulation_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/evolution_manager.hpp"
#include "gpu/gpu_batch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace moonai {

namespace {
constexpr float kMaxDensity = 10.0f;
constexpr float kMissingTargetSentinel = 2.0f;

Vec2 wrap_diff(Vec2 diff, float world_width, float world_height) {
  if (std::abs(diff.x) > world_width * 0.5f) {
    diff.x = diff.x > 0.0f ? diff.x - world_width : diff.x + world_width;
  }
  if (std::abs(diff.y) > world_height * 0.5f) {
    diff.y = diff.y > 0.0f ? diff.y - world_height : diff.y + world_height;
  }
  return diff;
}

float wrap_coord(float value, float limit) {
  while (value < 0.0f) {
    value += limit;
  }
  while (value >= limit) {
    value -= limit;
  }
  return value;
}

class DenseReproductionGrid {
public:
  DenseReproductionGrid(float world_width, float world_height, float cell_size,
                        std::size_t entity_count)
      : cell_size_(std::max(cell_size, 1.0f)),
        cols_(
            std::max(1, static_cast<int>(std::ceil(world_width / cell_size_)))),
        rows_(std::max(1,
                       static_cast<int>(std::ceil(world_height / cell_size_)))),
        counts_(static_cast<std::size_t>(cols_ * rows_), 0),
        offsets_(static_cast<std::size_t>(cols_ * rows_) + 1, 0),
        write_offsets_(static_cast<std::size_t>(cols_ * rows_), 0),
        entries_(entity_count, INVALID_ENTITY) {}

  void build(const PositionSoA &positions, std::size_t entity_count) {
    std::fill(counts_.begin(), counts_.end(), 0);
    std::fill(offsets_.begin(), offsets_.end(), 0);

    for (std::size_t idx = 0; idx < entity_count; ++idx) {
      const int cell = cell_index(positions.x[idx], positions.y[idx]);
      counts_[static_cast<std::size_t>(cell)] += 1;
    }

    for (std::size_t cell = 0; cell < counts_.size(); ++cell) {
      offsets_[cell + 1] = offsets_[cell] + counts_[cell];
    }

    std::copy(offsets_.begin(), offsets_.end() - 1, write_offsets_.begin());
    for (std::size_t idx = 0; idx < entity_count; ++idx) {
      const int cell = cell_index(positions.x[idx], positions.y[idx]);
      const std::size_t slot = static_cast<std::size_t>(
          write_offsets_[static_cast<std::size_t>(cell)]++);
      entries_[slot] = Entity{static_cast<uint32_t>(idx)};
    }
  }

  template <typename Callback>
  void for_each_candidate(Vec2 center, float radius,
                          Callback &&callback) const {
    const int cells_to_check =
        std::max(1, static_cast<int>(std::ceil(radius / cell_size_)));
    const int base_x = cell_coord(center.x, cols_);
    const int base_y = cell_coord(center.y, rows_);

    for (int dy = -cells_to_check; dy <= cells_to_check; ++dy) {
      for (int dx = -cells_to_check; dx <= cells_to_check; ++dx) {
        const int cell = flat_index(wrap_cell(base_x + dx, cols_),
                                    wrap_cell(base_y + dy, rows_));
        for (int slot = offsets_[static_cast<std::size_t>(cell)];
             slot < offsets_[static_cast<std::size_t>(cell) + 1]; ++slot) {
          callback(entries_[static_cast<std::size_t>(slot)]);
        }
      }
    }
  }

private:
  int cell_coord(float value, int limit) const {
    int coord = static_cast<int>(value / cell_size_);
    if (coord < 0) {
      return 0;
    }
    if (coord >= limit) {
      return limit - 1;
    }
    return coord;
  }

  int wrap_cell(int coord, int limit) const {
    coord %= limit;
    if (coord < 0) {
      coord += limit;
    }
    return coord;
  }

  int flat_index(int x, int y) const {
    return y * cols_ + x;
  }

  int cell_index(float x, float y) const {
    return flat_index(cell_coord(x, cols_), cell_coord(y, rows_));
  }

  float cell_size_;
  int cols_;
  int rows_;
  std::vector<int> counts_;
  std::vector<int> offsets_;
  std::vector<int> write_offsets_;
  std::vector<Entity> entries_;
};

void build_sensors(Registry &registry, const FoodStore &food_store,
                   const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);
  const float vision = config.vision_range;
  const float vision_sq = vision * vision;
  const auto agent_count = registry.size();
  const auto &positions = registry.positions();
  const auto &motion = registry.motion();
  const auto &vitals = registry.vitals();
  const auto &identity = registry.identity();
  auto &sensors = registry.sensors();

  for (std::size_t i = 0; i < agent_count; ++i) {
    float *sensor_ptr = sensors.input_ptr(i);
    sensor_ptr[0] = kMissingTargetSentinel;
    sensor_ptr[1] = kMissingTargetSentinel;
    sensor_ptr[2] = kMissingTargetSentinel;
    sensor_ptr[3] = kMissingTargetSentinel;
    sensor_ptr[4] = kMissingTargetSentinel;
    sensor_ptr[5] = kMissingTargetSentinel;
    sensor_ptr[6] = 0.0f;
    sensor_ptr[7] = 0.0f;
    sensor_ptr[8] = 0.0f;
    sensor_ptr[9] = 0.0f;
    sensor_ptr[10] = 0.0f;
    sensor_ptr[11] = 0.0f;

    if (!vitals.alive[i]) {
      continue;
    }

    const Vec2 pos{positions.x[i], positions.y[i]};
    float nearest_pred_dist_sq = std::numeric_limits<float>::max();
    float nearest_prey_dist_sq = std::numeric_limits<float>::max();
    float nearest_food_dist_sq = std::numeric_limits<float>::max();
    Vec2 nearest_pred_dir{0.0f, 0.0f};
    Vec2 nearest_prey_dir{0.0f, 0.0f};
    Vec2 nearest_food_dir{0.0f, 0.0f};
    int local_predators = 0;
    int local_prey = 0;
    int local_food = 0;

    for (std::size_t other = 0; other < agent_count; ++other) {
      if (other == i || !vitals.alive[other]) {
        continue;
      }

      Vec2 diff =
          wrap_diff({positions.x[other] - pos.x, positions.y[other] - pos.y},
                    world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq > vision_sq || dist_sq <= 0.0f) {
        continue;
      }

      if (identity.type[other] == IdentitySoA::TYPE_PREDATOR) {
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

    for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
      if (!food_store.active()[food_idx]) {
        continue;
      }

      Vec2 diff = wrap_diff({food_store.pos_x()[food_idx] - pos.x,
                             food_store.pos_y()[food_idx] - pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq > vision_sq) {
        continue;
      }

      ++local_food;
      if (dist_sq < nearest_food_dist_sq) {
        nearest_food_dist_sq = dist_sq;
        nearest_food_dir = diff;
      }
    }

    if (nearest_pred_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[0] = std::clamp(nearest_pred_dir.x / vision, -1.0f, 1.0f);
      sensor_ptr[1] = std::clamp(nearest_pred_dir.y / vision, -1.0f, 1.0f);
    }
    if (nearest_prey_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[2] = std::clamp(nearest_prey_dir.x / vision, -1.0f, 1.0f);
      sensor_ptr[3] = std::clamp(nearest_prey_dir.y / vision, -1.0f, 1.0f);
    }
    if (nearest_food_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[4] = std::clamp(nearest_food_dir.x / vision, -1.0f, 1.0f);
      sensor_ptr[5] = std::clamp(nearest_food_dir.y / vision, -1.0f, 1.0f);
    }

    sensor_ptr[6] = std::clamp(
        vitals.energy[i] / (static_cast<float>(config.initial_energy) * 2.0f),
        0.0f, 1.0f);
    if (motion.speed[i] > 0.0f) {
      sensor_ptr[7] =
          std::clamp(motion.vel_x[i] / motion.speed[i], -1.0f, 1.0f);
      sensor_ptr[8] =
          std::clamp(motion.vel_y[i] / motion.speed[i], -1.0f, 1.0f);
    }
    sensor_ptr[9] = std::clamp(
        static_cast<float>(local_predators) / kMaxDensity, 0.0f, 1.0f);
    sensor_ptr[10] =
        std::clamp(static_cast<float>(local_prey) / kMaxDensity, 0.0f, 1.0f);
    sensor_ptr[11] =
        std::clamp(static_cast<float>(local_food) / kMaxDensity, 0.0f, 1.0f);
  }
}

void update_vitals(Registry &registry, const SimulationConfig &config) {
  auto &vitals = registry.vitals();

  for (std::size_t i = 0; i < registry.size(); ++i) {
    if (!vitals.alive[i]) {
      continue;
    }

    vitals.age[i] += 1;

    vitals.energy[i] -= config.energy_drain_per_step;
    const bool died_of_starvation = vitals.energy[i] <= 0.0f;
    const bool died_of_age =
        config.max_steps > 0 && vitals.age[i] >= config.max_steps;
    if (died_of_starvation || died_of_age) {
      vitals.energy[i] = 0.0f;
      vitals.alive[i] = 0;
    }
  }
}

void process_food(Registry &registry, FoodStore &food_store,
                  const SimulationConfig &config,
                  std::vector<int> &food_consumed_by) {
  std::fill(food_consumed_by.begin(), food_consumed_by.end(), -1);
  const float world_size = static_cast<float>(config.grid_size);
  const float range_sq = config.interaction_range * config.interaction_range;
  const auto &positions = registry.positions();
  auto &vitals = registry.vitals();
  const auto &identity = registry.identity();

  for (std::size_t prey_idx = 0; prey_idx < registry.size(); ++prey_idx) {
    if (!vitals.alive[prey_idx] ||
        identity.type[prey_idx] != IdentitySoA::TYPE_PREY) {
      continue;
    }

    int best_food = -1;
    float best_dist_sq = range_sq;
    const Vec2 prey_pos{positions.x[prey_idx], positions.y[prey_idx]};

    for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
      if (!food_store.active()[food_idx]) {
        continue;
      }

      Vec2 diff = wrap_diff({food_store.pos_x()[food_idx] - prey_pos.x,
                             food_store.pos_y()[food_idx] - prey_pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq <= best_dist_sq) {
        best_dist_sq = dist_sq;
        best_food = static_cast<int>(food_idx);
      }
    }

    if (best_food >= 0) {
      int &owner = food_consumed_by[static_cast<std::size_t>(best_food)];
      if (owner < 0 || static_cast<int>(prey_idx) < owner) {
        owner = static_cast<int>(prey_idx);
      }
    }
  }

  for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
    const int prey_idx = food_consumed_by[food_idx];
    if (!food_store.active()[food_idx] || prey_idx < 0 ||
        !vitals.alive[static_cast<std::size_t>(prey_idx)]) {
      continue;
    }

    food_store.active()[food_idx] = 0;
    vitals.energy[static_cast<std::size_t>(prey_idx)] +=
        static_cast<float>(config.energy_gain_from_food);
  }
}

void process_combat(Registry &registry, const SimulationConfig &config,
                    std::vector<int> &killed_by,
                    std::vector<uint32_t> &kill_counts) {
  std::fill(killed_by.begin(), killed_by.end(), -1);
  std::fill(kill_counts.begin(), kill_counts.end(), 0U);
  const float world_size = static_cast<float>(config.grid_size);
  const float range_sq = config.interaction_range * config.interaction_range;
  const auto &positions = registry.positions();
  auto &vitals = registry.vitals();
  const auto &identity = registry.identity();

  for (std::size_t predator_idx = 0; predator_idx < registry.size();
       ++predator_idx) {
    if (!vitals.alive[predator_idx] ||
        identity.type[predator_idx] != IdentitySoA::TYPE_PREDATOR) {
      continue;
    }

    int best_prey = -1;
    float best_dist_sq = range_sq;
    const Vec2 predator_pos{positions.x[predator_idx],
                            positions.y[predator_idx]};

    for (std::size_t prey_idx = 0; prey_idx < registry.size(); ++prey_idx) {
      if (!vitals.alive[prey_idx] ||
          identity.type[prey_idx] != IdentitySoA::TYPE_PREY) {
        continue;
      }

      Vec2 diff = wrap_diff({positions.x[prey_idx] - predator_pos.x,
                             positions.y[prey_idx] - predator_pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq <= best_dist_sq) {
        best_dist_sq = dist_sq;
        best_prey = static_cast<int>(prey_idx);
      }
    }

    if (best_prey >= 0) {
      int &killer = killed_by[static_cast<std::size_t>(best_prey)];
      if (killer < 0 || static_cast<int>(predator_idx) < killer) {
        killer = static_cast<int>(predator_idx);
      }
    }
  }

  for (std::size_t prey_idx = 0; prey_idx < registry.size(); ++prey_idx) {
    const int killer_idx = killed_by[prey_idx];
    if (!vitals.alive[prey_idx] ||
        identity.type[prey_idx] != IdentitySoA::TYPE_PREY || killer_idx < 0 ||
        !vitals.alive[static_cast<std::size_t>(killer_idx)]) {
      continue;
    }

    vitals.alive[prey_idx] = 0;
    vitals.energy[static_cast<std::size_t>(killer_idx)] +=
        static_cast<float>(config.energy_gain_from_kill);
    kill_counts[static_cast<std::size_t>(killer_idx)] += 1;
  }
}

void apply_movement(Registry &registry, const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);
  auto &positions = registry.positions();
  auto &motion = registry.motion();
  const auto &vitals = registry.vitals();
  auto &stats = registry.stats();
  const auto &brain = registry.brain();

  for (std::size_t i = 0; i < registry.size(); ++i) {
    if (!vitals.alive[i]) {
      continue;
    }

    float dx = brain.decision_x[i];
    float dy = brain.decision_y[i];
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len > 1e-6f) {
      dx /= len;
      dy /= len;
    } else {
      dx = 0.0f;
      dy = 0.0f;
    }

    motion.vel_x[i] = dx * motion.speed[i];
    motion.vel_y[i] = dy * motion.speed[i];

    const float old_x = positions.x[i];
    const float old_y = positions.y[i];

    positions.x[i] += motion.vel_x[i];
    positions.y[i] += motion.vel_y[i];

    while (positions.x[i] < 0.0f)
      positions.x[i] += world_size;
    while (positions.x[i] >= world_size)
      positions.x[i] -= world_size;
    while (positions.y[i] < 0.0f)
      positions.y[i] += world_size;
    while (positions.y[i] >= world_size)
      positions.y[i] -= world_size;

    Vec2 move = wrap_diff({positions.x[i] - old_x, positions.y[i] - old_y},
                          world_size, world_size);
    stats.distance_traveled[i] += std::sqrt(move.x * move.x + move.y * move.y);
  }
}

void collect_cpu_step_events(Registry &registry, const FoodStore &food_store,
                             const std::vector<uint8_t> &was_alive,
                             const std::vector<uint8_t> &was_food_active,
                             const std::vector<int> &food_consumed_by,
                             const std::vector<int> &killed_by,
                             const std::vector<uint32_t> &kill_counts,
                             std::vector<SimEvent> &events) {
  auto &stats = registry.stats();
  const auto &positions = registry.positions();
  const auto &vitals = registry.vitals();

  for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
    const int prey_idx = food_consumed_by[food_idx];
    if (!was_food_active[food_idx] || food_store.active()[food_idx] ||
        prey_idx < 0 || static_cast<std::size_t>(prey_idx) >= registry.size()) {
      continue;
    }

    stats.food_eaten[static_cast<std::size_t>(prey_idx)] += 1;
    events.push_back(SimEvent{
        SimEvent::Food, Entity{static_cast<uint32_t>(prey_idx)}, INVALID_ENTITY,
        Vec2{positions.x[static_cast<std::size_t>(prey_idx)],
             positions.y[static_cast<std::size_t>(prey_idx)]}});
  }

  for (std::size_t agent_idx = 0; agent_idx < registry.size(); ++agent_idx) {
    if (kill_counts[agent_idx] > 0) {
      stats.kills[agent_idx] += static_cast<int>(kill_counts[agent_idx]);
    }

    if (killed_by[agent_idx] >= 0 &&
        static_cast<std::size_t>(killed_by[agent_idx]) < registry.size()) {
      events.push_back(SimEvent{
          SimEvent::Kill, Entity{static_cast<uint32_t>(killed_by[agent_idx])},
          Entity{static_cast<uint32_t>(agent_idx)},
          Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }

    if (was_alive[agent_idx] && !vitals.alive[agent_idx]) {
      const Entity entity{static_cast<uint32_t>(agent_idx)};
      events.push_back(
          SimEvent{SimEvent::Death, entity, entity,
                   Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }
  }
}
} // namespace

SimulationManager::SimulationManager(const SimulationConfig &config)
    : config_(config) {}

void SimulationManager::initialize(AppState &state) {
  initialize(state, true);
}

void SimulationManager::initialize(AppState &state, bool log_initialization) {
  state.food_store.initialize(config_, state.runtime.rng);

  if (log_initialization) {
    spdlog::info("Simulation initialized: {} food pellets (seed: {})",
                 config_.food_count, state.runtime.rng.seed());
  }
}

void SimulationManager::collect_gpu_step_events(
    AppState &state, const std::vector<uint8_t> &was_alive,
    const std::vector<uint8_t> &was_food_active) {
  auto &stats = state.registry.stats();
  const auto &positions = state.registry.positions();

  for (std::size_t food_idx = 0; food_idx < state.food_store.size();
       ++food_idx) {
    const int prey_idx = gpu_batch_->buffer().host_food_consumed_by()[food_idx];
    if (!was_food_active[food_idx] || state.food_store.active()[food_idx] ||
        prey_idx < 0 ||
        static_cast<std::size_t>(prey_idx) >= state.registry.size()) {
      continue;
    }

    const Entity prey{static_cast<uint32_t>(prey_idx)};
    stats.food_eaten[prey_idx] += 1;
    state.runtime.last_step_events.push_back(
        SimEvent{SimEvent::Food, prey, INVALID_ENTITY,
                 Vec2{positions.x[prey_idx], positions.y[prey_idx]}});
  }

  for (std::size_t agent_idx = 0; agent_idx < state.registry.size();
       ++agent_idx) {
    if (gpu_batch_->buffer().host_agent_kill_counts()[agent_idx] > 0) {
      stats.kills[agent_idx] += static_cast<int>(
          gpu_batch_->buffer().host_agent_kill_counts()[agent_idx]);
    }

    const int killer_idx =
        gpu_batch_->buffer().host_agent_killed_by()[agent_idx];
    if (killer_idx >= 0 &&
        static_cast<std::size_t>(killer_idx) < state.registry.size()) {
      state.runtime.last_step_events.push_back(
          SimEvent{SimEvent::Kill, Entity{static_cast<uint32_t>(killer_idx)},
                   Entity{static_cast<uint32_t>(agent_idx)},
                   Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }

    if (was_alive[agent_idx] && state.registry.vitals().alive[agent_idx] == 0) {
      const Entity entity{static_cast<uint32_t>(agent_idx)};
      state.runtime.last_step_events.push_back(
          SimEvent{SimEvent::Death, entity, entity,
                   Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }
  }
}

void SimulationManager::compact_registry(AppState &state,
                                         EvolutionManager &evolution) {
  const auto result = state.registry.compact_dead();
  for (const auto &[from, to] : result.moved) {
    evolution.on_entity_moved(state, from, to);
  }
  for (Entity removed : result.removed) {
    evolution.on_entity_destroyed(state, removed);
  }
}

void SimulationManager::refresh_world_state_after_step(AppState &state) {
  MOONAI_PROFILE_SCOPE("food_respawn");
  state.food_store.respawn_step(config_, state.runtime.step,
                                state.runtime.rng.seed());
}

void SimulationManager::step(AppState &state, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_cpu");

  state.runtime.last_step_events.clear();
  state.runtime.pending_offspring.clear();
  state.runtime.step_events.clear();

  const std::vector<uint8_t> was_alive = state.registry.vitals().alive;
  const std::vector<uint8_t> was_food_active = state.food_store.active();
  std::vector<int> food_consumed_by(state.food_store.size(), -1);
  std::vector<int> killed_by(state.registry.size(), -1);
  std::vector<uint32_t> kill_counts(state.registry.size(), 0U);

  build_sensors(state.registry, state.food_store, config_);
  evolution.compute_actions(state);
  update_vitals(state.registry, config_);
  process_food(state.registry, state.food_store, config_, food_consumed_by);
  process_combat(state.registry, config_, killed_by, kill_counts);
  apply_movement(state.registry, config_);

  collect_cpu_step_events(state.registry, state.food_store, was_alive,
                          was_food_active, food_consumed_by, killed_by,
                          kill_counts, state.runtime.last_step_events);
  compact_registry(state, evolution);
  refresh_world_state_after_step(state);
  state.runtime.pending_offspring = find_reproduction_pairs(state);

  for (const auto &event : state.runtime.last_step_events) {
    switch (event.type) {
      case SimEvent::Kill:
        ++state.runtime.step_events.kills;
        break;
      case SimEvent::Food:
        ++state.runtime.step_events.food_eaten;
        break;
      case SimEvent::Birth:
        ++state.runtime.step_events.births;
        break;
      case SimEvent::Death:
        ++state.runtime.step_events.deaths;
        break;
    }
  }
}

void SimulationManager::reset(AppState &state) {
  initialize(state, false);
}

std::vector<PendingOffspring>
SimulationManager::find_reproduction_pairs(const AppState &state) const {
  MOONAI_PROFILE_SCOPE("find_reproduction_pairs");
  std::vector<PendingOffspring> pairs;
  std::vector<uint8_t> used(state.registry.size(), 0);

  const auto &positions = state.registry.positions();
  const auto &vitals = state.registry.vitals();
  const auto &identity = state.registry.identity();
  DenseReproductionGrid grid(static_cast<float>(config_.grid_size),
                             static_cast<float>(config_.grid_size),
                             config_.mate_range, state.registry.size());
  grid.build(positions, state.registry.size());
  const float world_size = static_cast<float>(config_.grid_size);

  for (std::size_t idx = 0; idx < state.registry.size(); ++idx) {
    const Entity entity{static_cast<uint32_t>(idx)};
    if (vitals.energy[idx] < config_.reproduction_energy_threshold ||
        used[idx] != 0) {
      continue;
    }

    const Vec2 pos{positions.x[idx], positions.y[idx]};
    Entity best_mate = INVALID_ENTITY;
    float best_dist_sq = config_.mate_range * config_.mate_range;

    grid.for_each_candidate(pos, config_.mate_range, [&](Entity mate_id) {
      if (mate_id == entity || used[mate_id.index] != 0) {
        return;
      }

      const std::size_t mate_idx = state.registry.index_of(mate_id);
      if (identity.type[mate_idx] != identity.type[idx] ||
          vitals.energy[mate_idx] < config_.reproduction_energy_threshold) {
        return;
      }

      const Vec2 mate_pos{positions.x[mate_idx], positions.y[mate_idx]};
      const Vec2 diff = wrap_diff(mate_pos - pos, world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_mate = mate_id;
      }
    });

    if (best_mate != INVALID_ENTITY) {
      const std::size_t mate_idx = state.registry.index_of(best_mate);
      const Vec2 mate_pos{positions.x[mate_idx], positions.y[mate_idx]};
      const Vec2 diff = wrap_diff(mate_pos - pos, world_size, world_size);

      PendingOffspring pair;
      pair.parent_a = entity;
      pair.parent_b = best_mate;
      pair.spawn_position = {wrap_coord(pos.x + diff.x * 0.5f, world_size),
                             wrap_coord(pos.y + diff.y * 0.5f, world_size)};
      pairs.push_back(pair);
      used[idx] = 1;
      used[mate_idx] = 1;
    }
  }

  return pairs;
}

SimulationManager::~SimulationManager() = default;

void SimulationManager::ensure_gpu_capacity(std::size_t agent_count,
                                            std::size_t food_count) {
  if (!gpu_enabled_) {
    return;
  }

  const bool needs_batch = !gpu_batch_;
  const bool needs_resize =
      gpu_batch_ && (agent_count > gpu_batch_->agent_capacity() ||
                     food_count > gpu_batch_->food_capacity());
  if (!needs_batch && !needs_resize) {
    return;
  }

  const std::size_t current_agent_capacity =
      gpu_batch_ ? gpu_batch_->agent_capacity() : 0;
  const std::size_t current_food_capacity =
      gpu_batch_ ? gpu_batch_->food_capacity() : 0;
  const std::size_t new_agent_capacity = std::max(
      agent_count,
      current_agent_capacity == 0 ? agent_count : current_agent_capacity * 2);
  const std::size_t new_food_capacity =
      std::max(food_count,
               current_food_capacity == 0 ? food_count : current_food_capacity);

  gpu_batch_ =
      std::make_unique<gpu::GpuBatch>(new_agent_capacity, new_food_capacity);
  spdlog::info(
      "GPU batch processing enabled with capacities {} agents / {} food",
      new_agent_capacity, new_food_capacity);
}

void SimulationManager::enable_gpu(bool enable) {
  gpu_enabled_ = enable;
  if (enable) {
    ensure_gpu_capacity(
        static_cast<std::size_t>(config_.predator_count + config_.prey_count),
        static_cast<std::size_t>(config_.food_count));
  } else if (!enable) {
    gpu_batch_.reset();
    spdlog::info("GPU batch processing disabled");
  }
}

void SimulationManager::step_gpu(AppState &state, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_gpu");

  state.runtime.last_step_events.clear();
  state.runtime.pending_offspring.clear();
  state.runtime.step_events.clear();

  if (!gpu_batch_ || !gpu_batch_->ok()) {
    spdlog::error(
        "GPU batch not initialized or in error state, falling back to CPU");
    return step(state, evolution);
  }

  const std::size_t agent_count = state.registry.size();
  const std::size_t food_count = state.food_store.size();
  if (agent_count == 0) {
    return;
  }

  std::vector<uint8_t> was_alive;
  std::vector<uint8_t> was_food_active;
  {
    MOONAI_PROFILE_SCOPE("gpu_ensure_capacity");
    ensure_gpu_capacity(agent_count, food_count);
  }
  {
    MOONAI_PROFILE_SCOPE("gpu_pack_state");
    was_alive = state.registry.vitals().alive;
    was_food_active = state.food_store.active();

    auto &buffer = gpu_batch_->buffer();
    std::memcpy(buffer.host_agent_positions_x(),
                state.registry.positions().x.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_positions_y(),
                state.registry.positions().y.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_velocities_x(),
                state.registry.motion().vel_x.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_velocities_y(),
                state.registry.motion().vel_y.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_speed(), state.registry.motion().speed.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_energy(),
                state.registry.vitals().energy.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_age(), state.registry.vitals().age.data(),
                agent_count * sizeof(int));
    for (std::size_t i = 0; i < agent_count; ++i) {
      buffer.host_agent_alive()[i] = state.registry.vitals().alive[i];
      buffer.host_agent_types()[i] = state.registry.identity().type[i];
    }
    std::memcpy(buffer.host_agent_distance_traveled(),
                state.registry.stats().distance_traveled.data(),
                agent_count * sizeof(float));

    std::memcpy(buffer.host_food_positions_x(), state.food_store.pos_x().data(),
                food_count * sizeof(float));
    std::memcpy(buffer.host_food_positions_y(), state.food_store.pos_y().data(),
                food_count * sizeof(float));
    for (std::size_t i = 0; i < food_count; ++i) {
      buffer.host_food_active()[i] = state.food_store.active()[i];
      buffer.host_food_consumed_by()[i] = -1;
    }
  }

  gpu::GpuStepParams params;
  params.world_width = static_cast<float>(config_.grid_size);
  params.world_height = static_cast<float>(config_.grid_size);
  params.energy_drain_per_step = config_.energy_drain_per_step;
  params.vision_range = config_.vision_range;
  params.max_energy = static_cast<float>(config_.initial_energy);
  params.max_age = config_.max_steps;
  params.interaction_range = config_.interaction_range;
  params.energy_gain_from_food =
      static_cast<float>(config_.energy_gain_from_food);
  params.energy_gain_from_kill =
      static_cast<float>(config_.energy_gain_from_kill);

  {
    MOONAI_PROFILE_SCOPE("gpu_upload_enqueue");
    gpu_batch_->upload_async(agent_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_sensors");
    gpu_batch_->launch_build_sensors_async(params, agent_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_neural");
    if (!evolution.launch_gpu_neural(state, *gpu_batch_, agent_count)) {
      MOONAI_PROFILE_SCOPE("cpu_fallback");
      spdlog::error("GPU neural inference failed, disabling GPU path and "
                    "retrying on CPU");
      gpu_enabled_ = false;
      gpu_batch_.reset();
      return step(state, evolution);
    }
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_step");
    gpu_batch_->launch_post_inference_async(params, agent_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_download_enqueue");
    gpu_batch_->download_async(agent_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_synchronize");
    gpu_batch_->synchronize();
  }

  if (!gpu_batch_->ok()) {
    MOONAI_PROFILE_SCOPE("cpu_fallback");
    spdlog::error("GPU step failed, disabling GPU path and retrying on CPU");
    gpu_enabled_ = false;
    gpu_batch_.reset();
    return step(state, evolution);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_apply_results");
    {
      MOONAI_PROFILE_SCOPE("apply_dense_results");
      auto &buffer = gpu_batch_->buffer();
      std::memcpy(state.registry.positions().x.data(),
                  buffer.host_agent_positions_x(), agent_count * sizeof(float));
      std::memcpy(state.registry.positions().y.data(),
                  buffer.host_agent_positions_y(), agent_count * sizeof(float));
      std::memcpy(state.registry.motion().vel_x.data(),
                  buffer.host_agent_velocities_x(),
                  agent_count * sizeof(float));
      std::memcpy(state.registry.motion().vel_y.data(),
                  buffer.host_agent_velocities_y(),
                  agent_count * sizeof(float));
      std::memcpy(state.registry.vitals().energy.data(),
                  buffer.host_agent_energy(), agent_count * sizeof(float));
      std::memcpy(state.registry.vitals().age.data(), buffer.host_agent_age(),
                  agent_count * sizeof(int));
      std::memcpy(state.registry.stats().distance_traveled.data(),
                  buffer.host_agent_distance_traveled(),
                  agent_count * sizeof(float));
      std::memcpy(state.registry.sensors().inputs.data(),
                  buffer.host_agent_sensor_inputs(),
                  agent_count * SensorSoA::INPUT_COUNT * sizeof(float));
      for (std::size_t i = 0; i < agent_count; ++i) {
        state.registry.vitals().alive[i] =
            static_cast<uint8_t>(buffer.host_agent_alive()[i]);
        state.registry.brain().decision_x[i] =
            buffer.host_agent_brain_outputs()[i * SensorSoA::OUTPUT_COUNT];
        state.registry.brain().decision_y[i] =
            buffer.host_agent_brain_outputs()[i * SensorSoA::OUTPUT_COUNT + 1];
      }
      for (std::size_t i = 0; i < food_count; ++i) {
        state.food_store.pos_x()[i] = buffer.host_food_positions_x()[i];
        state.food_store.pos_y()[i] = buffer.host_food_positions_y()[i];
        state.food_store.active()[i] =
            static_cast<uint8_t>(buffer.host_food_active()[i]);
      }
    }
    {
      MOONAI_PROFILE_SCOPE("collect_step_events");
      collect_gpu_step_events(state, was_alive, was_food_active);
    }
  }
  {
    MOONAI_PROFILE_SCOPE("compact_registry");
    compact_registry(state, evolution);
  }
  {
    MOONAI_PROFILE_SCOPE("refresh_world_state");
    refresh_world_state_after_step(state);
  }
  {
    MOONAI_PROFILE_SCOPE("find_reproduction_pairs");
    state.runtime.pending_offspring = find_reproduction_pairs(state);
  }

  for (const auto &event : state.runtime.last_step_events) {
    switch (event.type) {
      case SimEvent::Kill:
        ++state.runtime.step_events.kills;
        break;
      case SimEvent::Food:
        ++state.runtime.step_events.food_eaten;
        break;
      case SimEvent::Birth:
        ++state.runtime.step_events.births;
        break;
      case SimEvent::Death:
        ++state.runtime.step_events.deaths;
        break;
    }
  }
}

} // namespace moonai
