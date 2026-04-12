#include "simulation/cpu.hpp"

#include "core/profiler_macros.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace moonai::simulation::cpu {

namespace {

constexpr float kMaxDensity = 10.0f;
constexpr float kMissingTargetSentinel = 2.0f;

void build_sensors(AgentRegistry &self_agents, const AgentRegistry &predator_agents, const AgentRegistry &prey_agents,
                   const Food &food_store, const SimulationConfig &config, float agent_speed,
                   std::vector<float> &sensors_out) {
  const float world_size = static_cast<float>(config.grid_size);
  const float vision = config.vision_range;
  const float vision_sq = vision * vision;
  const uint32_t self_count = static_cast<uint32_t>(self_agents.size());
  const bool predators_are_self = &self_agents == &predator_agents;
  const bool prey_are_self = &self_agents == &prey_agents;

  sensors_out.resize(self_count * SENSOR_COUNT);

  for (uint32_t i = 0; i < self_count; ++i) {
    float *sensor_ptr = &sensors_out[i * SENSOR_COUNT];
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
    sensor_ptr[12] = 0.0f; // bound_x
    sensor_ptr[13] = 0.0f; // bound_y

    if (!self_agents.alive[i]) {
      continue;
    }

    const Vec2 pos{self_agents.pos_x[i], self_agents.pos_y[i]};
    float nearest_pred_dist_sq = std::numeric_limits<float>::max();
    float nearest_prey_dist_sq = std::numeric_limits<float>::max();
    float nearest_food_dist_sq = std::numeric_limits<float>::max();
    Vec2 nearest_pred_dir{0.0f, 0.0f};
    Vec2 nearest_prey_dir{0.0f, 0.0f};
    Vec2 nearest_food_dir{0.0f, 0.0f};
    int local_predators = 0;
    int local_prey = 0;
    int local_food = 0;

    const uint32_t predator_count = static_cast<uint32_t>(predator_agents.size());
    for (uint32_t other = 0; other < predator_count; ++other) {
      if (!predator_agents.alive[other] || (predators_are_self && other == i)) {
        continue;
      }

      Vec2 diff{predator_agents.pos_x[other] - pos.x, predator_agents.pos_y[other] - pos.y};
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

      Vec2 diff{prey_agents.pos_x[other] - pos.x, prey_agents.pos_y[other] - pos.y};
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

      Vec2 diff{food_store.pos_x[food_idx] - pos.x, food_store.pos_y[food_idx] - pos.y};
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
      sensor_ptr[0] = nearest_pred_dir.x / vision;
      sensor_ptr[1] = nearest_pred_dir.y / vision;
    }
    if (nearest_prey_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[2] = nearest_prey_dir.x / vision;
      sensor_ptr[3] = nearest_prey_dir.y / vision;
    }
    if (nearest_food_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[4] = nearest_food_dir.x / vision;
      sensor_ptr[5] = nearest_food_dir.y / vision;
    }

    sensor_ptr[6] = std::clamp(self_agents.energy[i] / (static_cast<float>(config.initial_energy) * 2.0f), 0.0f, 1.0f);
    if (agent_speed > 0.0f) {
      sensor_ptr[7] = std::clamp(self_agents.vel_x[i] / agent_speed, -1.0f, 1.0f);
      sensor_ptr[8] = std::clamp(self_agents.vel_y[i] / agent_speed, -1.0f, 1.0f);
    }
    sensor_ptr[9] = std::clamp(static_cast<float>(local_predators) / kMaxDensity, 0.0f, 1.0f);
    sensor_ptr[10] = std::clamp(static_cast<float>(local_prey) / kMaxDensity, 0.0f, 1.0f);
    sensor_ptr[11] = std::clamp(static_cast<float>(local_food) / kMaxDensity, 0.0f, 1.0f);

    // Boundary sensors: bound_x and bound_y
    // Sensor 12 (bound_x): negative = approaching right wall, positive =
    // approaching left wall
    const float dist_left = pos.x;
    const float dist_right = world_size - pos.x;
    if (dist_left < vision) {
      sensor_ptr[12] = 1.0f - (dist_left / vision);
    } else if (dist_right < vision) {
      sensor_ptr[12] = -(1.0f - (dist_right / vision));
    }

    // Sensor 13 (bound_y): negative = approaching bottom wall, positive =
    // approaching top wall
    const float dist_top = pos.y;
    const float dist_bottom = world_size - pos.y;
    if (dist_top < vision) {
      sensor_ptr[13] = 1.0f - (dist_top / vision);
    } else if (dist_bottom < vision) {
      sensor_ptr[13] = -(1.0f - (dist_bottom / vision));
    }
  }
}

void update_vitals(AgentRegistry &agents, const SimulationConfig &config) {
  const uint32_t entity_count = static_cast<uint32_t>(agents.size());
  for (uint32_t i = 0; i < entity_count; ++i) {
    if (!agents.alive[i]) {
      continue;
    }

    agents.age[i] += 1;
    agents.energy[i] -= config.energy_drain_per_step;

    const bool died_of_starvation = agents.energy[i] <= 0.0f;
    const bool died_of_age = config.max_steps > 0 && agents.age[i] >= config.max_steps;
    if (died_of_starvation || died_of_age) {
      agents.energy[i] = 0.0f;
      agents.alive[i] = 0;
    }
  }
}

void process_food(AgentRegistry &prey_registry, Food &food_store, const SimulationConfig &config,
                  std::vector<int> &food_consumed_by) {
  std::fill(food_consumed_by.begin(), food_consumed_by.end(), -1);
  const float range_sq = config.interaction_range * config.interaction_range;
  const uint32_t prey_count = static_cast<uint32_t>(prey_registry.size());

  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    if (!prey_registry.alive[prey_idx]) {
      continue;
    }

    int best_food = -1;
    float best_dist_sq = range_sq;
    const Vec2 prey_pos{prey_registry.pos_x[prey_idx], prey_registry.pos_y[prey_idx]};

    for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
      if (!food_store.active[food_idx]) {
        continue;
      }

      Vec2 diff{food_store.pos_x[food_idx] - prey_pos.x, food_store.pos_y[food_idx] - prey_pos.y};
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
    if (!food_store.active[food_idx] || prey_idx < 0 || !prey_registry.alive[prey_idx]) {
      continue;
    }

    food_store.active[food_idx] = 0;
    prey_registry.energy[prey_idx] += static_cast<float>(config.energy_gain_from_food);
  }
}

void process_combat(AgentRegistry &predator_registry, AgentRegistry &prey_registry, const SimulationConfig &config,
                    std::vector<int> &killed_by, std::vector<uint32_t> &kill_counts) {
  std::fill(killed_by.begin(), killed_by.end(), -1);
  std::fill(kill_counts.begin(), kill_counts.end(), 0U);
  const float range_sq = config.interaction_range * config.interaction_range;
  const uint32_t predator_count = static_cast<uint32_t>(predator_registry.size());
  const uint32_t prey_count = static_cast<uint32_t>(prey_registry.size());

  for (uint32_t predator_idx = 0; predator_idx < predator_count; ++predator_idx) {
    if (!predator_registry.alive[predator_idx]) {
      continue;
    }

    int best_prey = -1;
    float best_dist_sq = range_sq;
    const Vec2 predator_pos{predator_registry.pos_x[predator_idx], predator_registry.pos_y[predator_idx]};

    for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
      if (!prey_registry.alive[prey_idx]) {
        continue;
      }

      Vec2 diff{prey_registry.pos_x[prey_idx] - predator_pos.x, prey_registry.pos_y[prey_idx] - predator_pos.y};
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
    if (!prey_registry.alive[prey_idx] || killer_idx < 0 || !predator_registry.alive[killer_idx]) {
      continue;
    }

    prey_registry.alive[prey_idx] = 0;
    predator_registry.energy[killer_idx] += static_cast<float>(config.energy_gain_from_kill);
    kill_counts[killer_idx] += 1;
  }
}

void apply_movement(AgentRegistry &agents, const SimulationConfig &config, float agent_speed,
                    const std::vector<float> &decisions) {
  const float world_size = static_cast<float>(config.grid_size);
  const uint32_t entity_count = static_cast<uint32_t>(agents.size());

  for (uint32_t i = 0; i < entity_count; ++i) {
    if (!agents.alive[i]) {
      continue;
    }

    float dx = decisions[i * OUTPUT_COUNT];
    float dy = decisions[i * OUTPUT_COUNT + 1];
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len > 1e-6f) {
      dx /= len;
      dy /= len;
    } else {
      dx = 0.0f;
      dy = 0.0f;
    }

    agents.vel_x[i] = dx * agent_speed;
    agents.vel_y[i] = dy * agent_speed;

    agents.pos_x[i] += agents.vel_x[i];
    agents.pos_y[i] += agents.vel_y[i];

    if (agents.pos_x[i] < 0.0f) {
      agents.pos_x[i] = 0.0f;
      agents.vel_x[i] = 0.0f;
    } else if (agents.pos_x[i] >= world_size) {
      agents.pos_x[i] = world_size;
      agents.vel_x[i] = 0.0f;
    }
    if (agents.pos_y[i] < 0.0f) {
      agents.pos_y[i] = 0.0f;
      agents.vel_y[i] = 0.0f;
    } else if (agents.pos_y[i] >= world_size) {
      agents.pos_y[i] = world_size;
      agents.vel_y[i] = 0.0f;
    }
  }
}

void collect_food_events(AgentRegistry &prey_registry, const Food &food_store,
                         const std::vector<uint8_t> &was_food_active, const std::vector<int> &food_consumed_by,
                         MetricsSnapshot &metrics) {
  for (std::size_t food_idx = 0; food_idx < food_store.size(); ++food_idx) {
    const int prey_idx = food_consumed_by[food_idx];
    if (!was_food_active[food_idx] || food_store.active[food_idx] || prey_idx < 0 ||
        static_cast<uint32_t>(prey_idx) >= prey_registry.size()) {
      continue;
    }

    prey_registry.consumption[prey_idx] += 1;
    ++metrics.food_eaten;
  }
}

void collect_combat_events(AgentRegistry &predator_registry, const AgentRegistry &prey_registry,
                           const std::vector<int> &killed_by, const std::vector<uint32_t> &kill_counts,
                           MetricsSnapshot &metrics) {
  const uint32_t predator_count = static_cast<uint32_t>(predator_registry.size());
  for (uint32_t predator_idx = 0; predator_idx < predator_count; ++predator_idx) {
    if (kill_counts[predator_idx] > 0) {
      predator_registry.consumption[predator_idx] += static_cast<int>(kill_counts[predator_idx]);
    }
  }

  const uint32_t prey_count = static_cast<uint32_t>(prey_registry.size());
  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    const int killer_idx = killed_by[prey_idx];
    if (killer_idx < 0 || static_cast<uint32_t>(killer_idx) >= predator_registry.size()) {
      continue;
    }

    ++metrics.kills;
  }
}

void collect_death_events(const AgentRegistry &registry, MetricsSnapshot &metrics) {
  const uint32_t entity_count = static_cast<uint32_t>(registry.size());
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    if (registry.alive[idx] == 0) {
      ++metrics.deaths;
    }
  }
}

} // namespace

bool prepare_step(AppState &state, const SimulationConfig &config) {
  MOONAI_PROFILE_SCOPE("simulation_cpu");

  state.step_buffers.was_food_active = state.food.active;
  state.step_buffers.food_consumed_by.assign(state.food.size(), -1);
  state.step_buffers.killed_by.assign(state.prey.size(), -1);
  state.step_buffers.kill_counts.assign(state.predator.size(), 0U);

  build_sensors(state.predator, state.predator, state.prey, state.food, config, config.predator_speed,
                state.step_buffers.predator_sensors);
  build_sensors(state.prey, state.predator, state.prey, state.food, config, config.prey_speed,
                state.step_buffers.prey_sensors);

  return true;
}

bool resolve_step(AppState &state, const SimulationConfig &config) {
  MOONAI_PROFILE_SCOPE("simulation_cpu_resolve");

  update_vitals(state.predator, config);
  update_vitals(state.prey, config);
  process_food(state.prey, state.food, config, state.step_buffers.food_consumed_by);
  process_combat(state.predator, state.prey, config, state.step_buffers.killed_by, state.step_buffers.kill_counts);
  apply_movement(state.predator, config, config.predator_speed, state.step_buffers.predator_decisions);
  apply_movement(state.prey, config, config.prey_speed, state.step_buffers.prey_decisions);

  collect_food_events(state.prey, state.food, state.step_buffers.was_food_active, state.step_buffers.food_consumed_by,
                      state.metrics.step_delta);
  collect_combat_events(state.predator, state.prey, state.step_buffers.killed_by, state.step_buffers.kill_counts,
                        state.metrics.step_delta);
  collect_death_events(state.predator, state.metrics.step_delta);
  collect_death_events(state.prey, state.metrics.step_delta);

  return true;
}

} // namespace moonai::simulation::cpu
