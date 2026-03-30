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

template <typename RegistryT, typename Callback>
void collect_death_events_impl(const RegistryT &registry,
                               const std::vector<uint8_t> &was_alive,
                               Callback &&callback) {
  const uint32_t entity_count = static_cast<uint32_t>(registry.size());
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    if (was_alive[idx] == 0 || registry.agents.alive[idx] != 0) {
      continue;
    }

    callback(idx);
  }
}

} // namespace

void build_sensors(AgentSoA &self_agents, const PositionSoA &self_positions,
                   const AgentSoA &predator_agents,
                   const PositionSoA &predator_positions,
                   const AgentSoA &prey_agents,
                   const PositionSoA &prey_positions,
                   const FoodStore &food_store,
                   const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);
  const float vision = config.vision_range;
  const float vision_sq = vision * vision;
  const uint32_t self_count = static_cast<uint32_t>(self_agents.size());
  const bool predators_are_self = &self_agents == &predator_agents;
  const bool prey_are_self = &self_agents == &prey_agents;

  for (uint32_t i = 0; i < self_count; ++i) {
    float *sensor_ptr = self_agents.input_ptr(i);
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

    if (!self_agents.alive[i]) {
      continue;
    }

    const Vec2 pos{self_positions.x[i], self_positions.y[i]};
    float nearest_pred_dist_sq = std::numeric_limits<float>::max();
    float nearest_prey_dist_sq = std::numeric_limits<float>::max();
    float nearest_food_dist_sq = std::numeric_limits<float>::max();
    Vec2 nearest_pred_dir{0.0f, 0.0f};
    Vec2 nearest_prey_dir{0.0f, 0.0f};
    Vec2 nearest_food_dir{0.0f, 0.0f};
    int local_predators = 0;
    int local_prey = 0;
    int local_food = 0;

    const uint32_t predator_count =
        static_cast<uint32_t>(predator_agents.size());
    for (uint32_t other = 0; other < predator_count; ++other) {
      if (!predator_agents.alive[other] || (predators_are_self && other == i)) {
        continue;
      }

      Vec2 diff = wrap_diff({predator_positions.x[other] - pos.x,
                             predator_positions.y[other] - pos.y},
                            world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq > vision_sq || dist_sq <= 0.0f) {
        continue;
      }

      ++local_predators;
      if (dist_sq < nearest_pred_dist_sq) {
        nearest_pred_dist_sq = dist_sq;
        nearest_pred_dir = diff;
      }
    }

    const uint32_t prey_count = static_cast<uint32_t>(prey_agents.size());
    for (uint32_t other = 0; other < prey_count; ++other) {
      if (!prey_agents.alive[other] || (prey_are_self && other == i)) {
        continue;
      }

      Vec2 diff = wrap_diff(
          {prey_positions.x[other] - pos.x, prey_positions.y[other] - pos.y},
          world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq > vision_sq || dist_sq <= 0.0f) {
        continue;
      }

      ++local_prey;
      if (dist_sq < nearest_prey_dist_sq) {
        nearest_prey_dist_sq = dist_sq;
        nearest_prey_dir = diff;
      }
    }

    for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
      if (!food_store.active[food_idx]) {
        continue;
      }

      Vec2 diff = wrap_diff({food_store.positions.x[food_idx] - pos.x,
                             food_store.positions.y[food_idx] - pos.y},
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

    sensor_ptr[6] =
        std::clamp(self_agents.energy[i] /
                       (static_cast<float>(config.initial_energy) * 2.0f),
                   0.0f, 1.0f);
    if (self_agents.speed[i] > 0.0f) {
      sensor_ptr[7] =
          std::clamp(self_agents.vel_x[i] / self_agents.speed[i], -1.0f, 1.0f);
      sensor_ptr[8] =
          std::clamp(self_agents.vel_y[i] / self_agents.speed[i], -1.0f, 1.0f);
    }
    sensor_ptr[9] = std::clamp(
        static_cast<float>(local_predators) / kMaxDensity, 0.0f, 1.0f);
    sensor_ptr[10] =
        std::clamp(static_cast<float>(local_prey) / kMaxDensity, 0.0f, 1.0f);
    sensor_ptr[11] =
        std::clamp(static_cast<float>(local_food) / kMaxDensity, 0.0f, 1.0f);
  }
}

void update_vitals(AgentSoA &agents, const SimulationConfig &config) {
  const uint32_t entity_count = static_cast<uint32_t>(agents.size());
  for (uint32_t i = 0; i < entity_count; ++i) {
    if (!agents.alive[i]) {
      continue;
    }

    agents.age[i] += 1;
    agents.energy[i] -= config.energy_drain_per_step;

    const bool died_of_starvation = agents.energy[i] <= 0.0f;
    const bool died_of_age =
        config.max_steps > 0 && agents.age[i] >= config.max_steps;
    if (died_of_starvation || died_of_age) {
      agents.energy[i] = 0.0f;
      agents.alive[i] = 0;
    }
  }
}

void process_food(PreyRegistry &prey_registry, FoodStore &food_store,
                  const SimulationConfig &config,
                  std::vector<int> &food_consumed_by) {
  std::fill(food_consumed_by.begin(), food_consumed_by.end(), -1);
  const float world_size = static_cast<float>(config.grid_size);
  const float range_sq = config.interaction_range * config.interaction_range;
  const uint32_t prey_count = static_cast<uint32_t>(prey_registry.size());

  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    if (!prey_registry.agents.alive[prey_idx]) {
      continue;
    }

    int best_food = -1;
    float best_dist_sq = range_sq;
    const Vec2 prey_pos{prey_registry.positions.x[prey_idx],
                        prey_registry.positions.y[prey_idx]};

    for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
      if (!food_store.active[food_idx]) {
        continue;
      }

      Vec2 diff = wrap_diff({food_store.positions.x[food_idx] - prey_pos.x,
                             food_store.positions.y[food_idx] - prey_pos.y},
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
        !prey_registry.agents.alive[prey_idx]) {
      continue;
    }

    food_store.active[food_idx] = 0;
    prey_registry.agents.energy[prey_idx] +=
        static_cast<float>(config.energy_gain_from_food);
  }
}

void process_combat(PredatorRegistry &predator_registry,
                    PreyRegistry &prey_registry, const SimulationConfig &config,
                    std::vector<int> &killed_by,
                    std::vector<uint32_t> &kill_counts) {
  std::fill(killed_by.begin(), killed_by.end(), -1);
  std::fill(kill_counts.begin(), kill_counts.end(), 0U);
  const float world_size = static_cast<float>(config.grid_size);
  const float range_sq = config.interaction_range * config.interaction_range;
  const uint32_t predator_count =
      static_cast<uint32_t>(predator_registry.size());
  const uint32_t prey_count = static_cast<uint32_t>(prey_registry.size());

  for (uint32_t predator_idx = 0; predator_idx < predator_count;
       ++predator_idx) {
    if (!predator_registry.agents.alive[predator_idx]) {
      continue;
    }

    int best_prey = -1;
    float best_dist_sq = range_sq;
    const Vec2 predator_pos{predator_registry.positions.x[predator_idx],
                            predator_registry.positions.y[predator_idx]};

    for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
      if (!prey_registry.agents.alive[prey_idx]) {
        continue;
      }

      Vec2 diff =
          wrap_diff({prey_registry.positions.x[prey_idx] - predator_pos.x,
                     prey_registry.positions.y[prey_idx] - predator_pos.y},
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

  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    const int killer_idx = killed_by[prey_idx];
    if (!prey_registry.agents.alive[prey_idx] || killer_idx < 0 ||
        !predator_registry.agents.alive[killer_idx]) {
      continue;
    }

    prey_registry.agents.alive[prey_idx] = 0;
    predator_registry.agents.energy[killer_idx] +=
        static_cast<float>(config.energy_gain_from_kill);
    kill_counts[killer_idx] += 1;
  }
}

void apply_movement(PositionSoA &positions, AgentSoA &agents,
                    const SimulationConfig &config) {
  const float world_size = static_cast<float>(config.grid_size);
  const uint32_t entity_count = static_cast<uint32_t>(agents.size());

  for (uint32_t i = 0; i < entity_count; ++i) {
    if (!agents.alive[i]) {
      continue;
    }

    float dx = agents.decision_x[i];
    float dy = agents.decision_y[i];
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len > 1e-6f) {
      dx /= len;
      dy /= len;
    } else {
      dx = 0.0f;
      dy = 0.0f;
    }

    agents.vel_x[i] = dx * agents.speed[i];
    agents.vel_y[i] = dy * agents.speed[i];

    const float old_x = positions.x[i];
    const float old_y = positions.y[i];

    positions.x[i] += agents.vel_x[i];
    positions.y[i] += agents.vel_y[i];

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
    agents.distance_traveled[i] += std::sqrt(move.x * move.x + move.y * move.y);
  }
}

void collect_food_events(PreyRegistry &prey_registry,
                         const FoodStore &food_store,
                         const std::vector<uint8_t> &was_food_active,
                         const std::vector<int> &food_consumed_by,
                         std::vector<SimEvent> &events) {
  for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
    const int prey_idx = food_consumed_by[food_idx];
    if (!was_food_active[food_idx] || food_store.active[food_idx] ||
        prey_idx < 0 ||
        static_cast<uint32_t>(prey_idx) >= prey_registry.size()) {
      continue;
    }

    prey_registry.prey.food_eaten[prey_idx] += 1;
    events.push_back(SimEvent{SimEvent::Food,
                              prey_registry.agents.entity_id[prey_idx], 0,
                              Vec2{prey_registry.positions.x[prey_idx],
                                   prey_registry.positions.y[prey_idx]}});
  }
}

void collect_combat_events(PredatorRegistry &predator_registry,
                           const PreyRegistry &prey_registry,
                           const std::vector<int> &killed_by,
                           const std::vector<uint32_t> &kill_counts,
                           std::vector<SimEvent> &events) {
  const uint32_t predator_count =
      static_cast<uint32_t>(predator_registry.size());
  for (uint32_t predator_idx = 0; predator_idx < predator_count;
       ++predator_idx) {
    if (kill_counts[predator_idx] > 0) {
      predator_registry.predator.kills[predator_idx] +=
          static_cast<int>(kill_counts[predator_idx]);
    }
  }

  const uint32_t prey_count = static_cast<uint32_t>(prey_registry.size());
  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    const int killer_idx = killed_by[prey_idx];
    if (killer_idx < 0 ||
        static_cast<uint32_t>(killer_idx) >= predator_registry.size()) {
      continue;
    }

    events.push_back(SimEvent{SimEvent::Kill,
                              predator_registry.agents.entity_id[killer_idx],
                              prey_registry.agents.entity_id[prey_idx],
                              Vec2{prey_registry.positions.x[prey_idx],
                                   prey_registry.positions.y[prey_idx]}});
  }
}

void collect_death_events(const PredatorRegistry &predator_registry,
                          const std::vector<uint8_t> &was_alive,
                          std::vector<SimEvent> &events) {
  collect_death_events_impl(predator_registry, was_alive, [&](uint32_t idx) {
    events.push_back(SimEvent{SimEvent::Death,
                              predator_registry.agents.entity_id[idx],
                              predator_registry.agents.entity_id[idx],
                              Vec2{predator_registry.positions.x[idx],
                                   predator_registry.positions.y[idx]}});
  });
}

void collect_death_events(const PreyRegistry &prey_registry,
                          const std::vector<uint8_t> &was_alive,
                          std::vector<SimEvent> &events) {
  collect_death_events_impl(prey_registry, was_alive, [&](uint32_t idx) {
    events.push_back(SimEvent{
        SimEvent::Death, prey_registry.agents.entity_id[idx],
        prey_registry.agents.entity_id[idx],
        Vec2{prey_registry.positions.x[idx], prey_registry.positions.y[idx]}});
  });
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
