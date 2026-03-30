#include "simulation/simulation_step_systems.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace moonai::simulation_detail {

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

} // namespace

void build_sensors(Registry &registry, const FoodStore &food_store,
                   const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);
  const float vision = config.vision_range;
  const float vision_sq = vision * vision;
  const uint32_t agent_count = static_cast<uint32_t>(registry.size());
  const auto &positions = registry.positions;
  const auto &motion = registry.motion;
  const auto &vitals = registry.vitals;
  const auto &identity = registry.identity;
  auto &sensors = registry.sensors;

  for (uint32_t i = 0; i < agent_count; ++i) {
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

    for (uint32_t other = 0; other < agent_count; ++other) {
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
      if (!food_store.active[food_idx]) {
        continue;
      }

      Vec2 diff = wrap_diff({food_store.pos_x[food_idx] - pos.x,
                             food_store.pos_y[food_idx] - pos.y},
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
  auto &vitals = registry.vitals;

  const uint32_t entity_count = static_cast<uint32_t>(registry.size());
  for (uint32_t i = 0; i < entity_count; ++i) {
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
  const auto &positions = registry.positions;
  auto &vitals = registry.vitals;
  const auto &identity = registry.identity;

  const uint32_t entity_count = static_cast<uint32_t>(registry.size());
  for (uint32_t prey_idx = 0; prey_idx < entity_count; ++prey_idx) {
    if (!vitals.alive[prey_idx] ||
        identity.type[prey_idx] != IdentitySoA::TYPE_PREY) {
      continue;
    }

    int best_food = -1;
    float best_dist_sq = range_sq;
    const Vec2 prey_pos{positions.x[prey_idx], positions.y[prey_idx]};

    for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
      if (!food_store.active[food_idx]) {
        continue;
      }

      Vec2 diff = wrap_diff({food_store.pos_x[food_idx] - prey_pos.x,
                             food_store.pos_y[food_idx] - prey_pos.y},
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
    if (!food_store.active[food_idx] || prey_idx < 0 ||
        !vitals.alive[prey_idx]) {
      continue;
    }

    food_store.active[food_idx] = 0;
    vitals.energy[prey_idx] += static_cast<float>(config.energy_gain_from_food);
  }
}

void process_combat(Registry &registry, const SimulationConfig &config,
                    std::vector<int> &killed_by,
                    std::vector<uint32_t> &kill_counts) {
  std::fill(killed_by.begin(), killed_by.end(), -1);
  std::fill(kill_counts.begin(), kill_counts.end(), 0U);
  const float world_size = static_cast<float>(config.grid_size);
  const float range_sq = config.interaction_range * config.interaction_range;
  const auto &positions = registry.positions;
  auto &vitals = registry.vitals;
  const auto &identity = registry.identity;

  const uint32_t entity_count = static_cast<uint32_t>(registry.size());
  for (uint32_t predator_idx = 0; predator_idx < entity_count; ++predator_idx) {
    if (!vitals.alive[predator_idx] ||
        identity.type[predator_idx] != IdentitySoA::TYPE_PREDATOR) {
      continue;
    }

    int best_prey = -1;
    float best_dist_sq = range_sq;
    const Vec2 predator_pos{positions.x[predator_idx],
                            positions.y[predator_idx]};

    for (uint32_t prey_idx = 0; prey_idx < entity_count; ++prey_idx) {
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

  for (uint32_t prey_idx = 0; prey_idx < entity_count; ++prey_idx) {
    const int killer_idx = killed_by[prey_idx];
    if (!vitals.alive[prey_idx] ||
        identity.type[prey_idx] != IdentitySoA::TYPE_PREY || killer_idx < 0 ||
        !vitals.alive[killer_idx]) {
      continue;
    }

    vitals.alive[prey_idx] = 0;
    vitals.energy[killer_idx] +=
        static_cast<float>(config.energy_gain_from_kill);
    kill_counts[killer_idx] += 1;
  }
}

void apply_movement(Registry &registry, const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);
  auto &positions = registry.positions;
  auto &motion = registry.motion;
  const auto &vitals = registry.vitals;
  auto &stats = registry.stats;
  const auto &brain = registry.brain;

  const uint32_t entity_count = static_cast<uint32_t>(registry.size());
  for (uint32_t i = 0; i < entity_count; ++i) {
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
  auto &stats = registry.stats;
  const auto &positions = registry.positions;
  const auto &vitals = registry.vitals;

  for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
    const int prey_idx = food_consumed_by[food_idx];
    if (!was_food_active[food_idx] || food_store.active[food_idx] ||
        prey_idx < 0 || static_cast<uint32_t>(prey_idx) >= registry.size()) {
      continue;
    }

    stats.food_eaten[prey_idx] += 1;
    events.push_back(
        SimEvent{SimEvent::Food, static_cast<uint32_t>(prey_idx), INVALID_ENTITY,
                 Vec2{positions.x[prey_idx], positions.y[prey_idx]}});
  }

  const uint32_t entity_count = static_cast<uint32_t>(registry.size());
  for (uint32_t agent_idx = 0; agent_idx < entity_count; ++agent_idx) {
    if (kill_counts[agent_idx] > 0) {
      stats.kills[agent_idx] += static_cast<int>(kill_counts[agent_idx]);
    }

    if (killed_by[agent_idx] >= 0 &&
        static_cast<uint32_t>(killed_by[agent_idx]) < registry.size()) {
      events.push_back(SimEvent{
          SimEvent::Kill, static_cast<uint32_t>(killed_by[agent_idx]), agent_idx,
          Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }

    if (was_alive[agent_idx] && !vitals.alive[agent_idx]) {
      const uint32_t entity = agent_idx;
      events.push_back(
          SimEvent{SimEvent::Death, entity, entity,
                   Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }
  }
}

void accumulate_events(EventCounters &counters,
                       const std::vector<SimEvent> &events) {
  for (const auto &event : events) {
    switch (event.type) {
      case SimEvent::Kill:
        ++counters.kills;
        break;
      case SimEvent::Food:
        ++counters.food_eaten;
        break;
      case SimEvent::Birth:
        ++counters.births;
        break;
      case SimEvent::Death:
        ++counters.deaths;
        break;
    }
  }
}

} // namespace moonai::simulation_detail
