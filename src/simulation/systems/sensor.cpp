#include "simulation/systems/sensor.hpp"
#include "simulation/components.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace moonai {

SensorSystem::SensorSystem(const SpatialGridECS &agent_grid, float world_width,
                           float world_height, float max_energy, bool has_walls)
    : agent_grid_(agent_grid), world_width_(world_width),
      world_height_(world_height), max_energy_(max_energy),
      has_walls_(has_walls) {}

void SensorSystem::update(Registry &registry, float dt) {
  auto &positions = registry.positions();
  auto &vitals = registry.vitals();
  auto &identity = registry.identity();
  auto &motion = registry.motion();
  auto &sensors = registry.sensors();

  const size_t count = registry.size();

  for (size_t i = 0; i < count; ++i) {
    if (!vitals.alive[i]) {
      continue;
    }

    build_sensors_for_entity(i, registry);
  }
}

void SensorSystem::build_sensors_for_entity(size_t entity_idx,
                                            Registry &registry) {
  auto &positions = registry.positions();
  auto &vitals = registry.vitals();
  auto &identity = registry.identity();
  auto &motion = registry.motion();
  auto &sensors = registry.sensors();

  Vec2 pos{positions.x[entity_idx], positions.y[entity_idx]};
  float vision = VISION_RANGE;
  float vision_sq = vision * vision;
  uint8_t my_type = identity.type[entity_idx];

  // Initialize sensor inputs to defaults
  float *sensor_ptr = sensors.input_ptr(entity_idx);

  // Default values: -1 for distances (none in range), 0 for angles
  sensor_ptr[0] = -1.0f; // nearest_predator_dist
  sensor_ptr[1] = 0.0f;  // nearest_predator_angle
  sensor_ptr[2] = -1.0f; // nearest_prey_dist
  sensor_ptr[3] = 0.0f;  // nearest_prey_angle
  sensor_ptr[4] = -1.0f; // nearest_food_dist
  sensor_ptr[5] = 0.0f;  // nearest_food_angle
  sensor_ptr[6] = 1.0f;  // energy_level
  sensor_ptr[7] = 0.0f;  // speed_x
  sensor_ptr[8] = 0.0f;  // speed_y
  sensor_ptr[9] = 0.0f;  // local_predator_density
  sensor_ptr[10] = 0.0f; // local_prey_density
  sensor_ptr[11] = 1.0f; // wall_left
  sensor_ptr[12] = 1.0f; // wall_right
  sensor_ptr[13] = 1.0f; // wall_top
  sensor_ptr[14] = 1.0f; // wall_bottom

  // Query nearby entities
  auto nearby = agent_grid_.query_radius(pos, vision);

  float nearest_pred_dist_sq = std::numeric_limits<float>::max();
  float nearest_prey_dist_sq = std::numeric_limits<float>::max();
  Vec2 nearest_pred_dir{0.0f, 0.0f};
  Vec2 nearest_prey_dir{0.0f, 0.0f};
  int local_predators = 0;
  int local_prey = 0;

  // Get entity list for looking up component data
  const auto &living = registry.living_entities();

  for (Entity other_e : nearby) {
    // Skip self
    if (other_e.index == registry.living_entities()[entity_idx].index) {
      continue;
    }

    size_t other_idx = registry.index_of(other_e);
    if (other_idx == std::numeric_limits<size_t>::max()) {
      continue;
    }

    if (!registry.vitals().alive[other_idx]) {
      continue;
    }

    Vec2 other_pos{positions.x[other_idx], positions.y[other_idx]};
    Vec2 diff = wrap_diff({other_pos.x - pos.x, other_pos.y - pos.y});
    float dist_sq = diff.x * diff.x + diff.y * diff.y;

    if (dist_sq > vision_sq) {
      continue;
    }

    uint8_t other_type = identity.type[other_idx];

    if (other_type == IdentitySoA::TYPE_PREDATOR) {
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

  // Nearest predator
  if (nearest_pred_dist_sq < std::numeric_limits<float>::max()) {
    sensor_ptr[0] = std::sqrt(nearest_pred_dist_sq) / vision;
    sensor_ptr[1] = normalize_angle(nearest_pred_dir.x, nearest_pred_dir.y);
  }

  // Nearest prey
  if (nearest_prey_dist_sq < std::numeric_limits<float>::max()) {
    sensor_ptr[2] = std::sqrt(nearest_prey_dist_sq) / vision;
    sensor_ptr[3] = normalize_angle(nearest_prey_dir.x, nearest_prey_dir.y);
  }

  // Nearest food (for prey only) - query from spatial grid
  if (my_type == IdentitySoA::TYPE_PREY) {
    float nearest_food_dist_sq = std::numeric_limits<float>::max();
    Vec2 nearest_food_dir{0.0f, 0.0f};
    auto &food_state = registry.food_state();

    // Re-query spatial grid for food entities
    for (Entity other_e : nearby) {
      size_t other_idx = registry.index_of(other_e);
      if (other_idx == std::numeric_limits<size_t>::max()) {
        continue;
      }

      // Only consider food entities that are active
      if (identity.type[other_idx] != IdentitySoA::TYPE_FOOD ||
          !food_state.active[other_idx]) {
        continue;
      }

      Vec2 other_pos{positions.x[other_idx], positions.y[other_idx]};
      Vec2 diff = wrap_diff({other_pos.x - pos.x, other_pos.y - pos.y});
      float dist_sq = diff.x * diff.x + diff.y * diff.y;

      if (dist_sq < nearest_food_dist_sq) {
        nearest_food_dist_sq = dist_sq;
        nearest_food_dir = diff;
      }
    }

    if (nearest_food_dist_sq < std::numeric_limits<float>::max()) {
      sensor_ptr[4] = std::sqrt(nearest_food_dist_sq) / vision;
      sensor_ptr[5] = normalize_angle(nearest_food_dir.x, nearest_food_dir.y);
    }
  }

  // Self state
  sensor_ptr[6] =
      std::clamp(vitals.energy[entity_idx] / (max_energy_ * 2.0f), 0.0f, 1.0f);
  float speed = motion.speed[entity_idx];
  if (speed > 0.0f) {
    sensor_ptr[7] = std::clamp(motion.vel_x[entity_idx] / speed, -1.0f, 1.0f);
    sensor_ptr[8] = std::clamp(motion.vel_y[entity_idx] / speed, -1.0f, 1.0f);
  }

  // Density
  sensor_ptr[9] =
      std::clamp(static_cast<float>(local_predators) / MAX_DENSITY, 0.0f, 1.0f);
  sensor_ptr[10] =
      std::clamp(static_cast<float>(local_prey) / MAX_DENSITY, 0.0f, 1.0f);

  // Wall proximity (only in wall mode)
  if (has_walls_) {
    sensor_ptr[11] = std::clamp(pos.x / vision, 0.0f, 1.0f);
    sensor_ptr[12] = std::clamp((world_width_ - pos.x) / vision, 0.0f, 1.0f);
    sensor_ptr[13] = std::clamp(pos.y / vision, 0.0f, 1.0f);
    sensor_ptr[14] = std::clamp((world_height_ - pos.y) / vision, 0.0f, 1.0f);
  }
}

Vec2 SensorSystem::wrap_diff(Vec2 diff) const {
  if (!has_walls_) {
    if (std::abs(diff.x) > world_width_ * 0.5f) {
      diff.x = (diff.x > 0.0f) ? diff.x - world_width_ : diff.x + world_width_;
    }
    if (std::abs(diff.y) > world_height_ * 0.5f) {
      diff.y =
          (diff.y > 0.0f) ? diff.y - world_height_ : diff.y + world_height_;
    }
  }
  return diff;
}

float SensorSystem::normalize_angle(float dx, float dy) const {
  return std::atan2(dy, dx) / 3.14159265f; // maps to [-1, 1]
}

} // namespace moonai