#include "simulation/systems/movement.hpp"
#include "simulation/components.hpp"
#include <algorithm>
#include <cmath>

namespace moonai {

MovementSystem::MovementSystem(float world_width, float world_height,
                               bool has_walls)
    : world_width_(world_width), world_height_(world_height),
      has_walls_(has_walls) {}

void MovementSystem::update(Registry &registry) {
  auto &positions = registry.positions();
  auto &motion = registry.motion();
  auto &vitals = registry.vitals();
  auto &brain = registry.brain();
  auto &stats = registry.stats();

  const size_t count = registry.size();

  for (size_t i = 0; i < count; ++i) {
    if (!vitals.alive[i]) {
      continue;
    }

    float dx = brain.decision_x[i];
    float dy = brain.decision_y[i];

    float len = std::sqrt(dx * dx + dy * dy);
    if (len > 1e-6f) {
      dx /= len;
      dy /= len;
    } else {
      dx = 0.0f;
      dy = 0.0f;
    }

    float speed = motion.speed[i];
    motion.vel_x[i] = dx * speed;
    motion.vel_y[i] = dy * speed;

    float old_x = positions.x[i];
    float old_y = positions.y[i];

    positions.x[i] += motion.vel_x[i];
    positions.y[i] += motion.vel_y[i];

    apply_boundary(positions.x[i], positions.y[i]);

    float dx_pos = positions.x[i] - old_x;
    float dy_pos = positions.y[i] - old_y;

    if (!has_walls_) {
      if (std::abs(dx_pos) > world_width_ * 0.5f) {
        dx_pos =
            (dx_pos > 0.0f) ? dx_pos - world_width_ : dx_pos + world_width_;
      }
      if (std::abs(dy_pos) > world_height_ * 0.5f) {
        dy_pos =
            (dy_pos > 0.0f) ? dy_pos - world_height_ : dy_pos + world_height_;
      }
    }

    stats.distance_traveled[i] += std::sqrt(dx_pos * dx_pos + dy_pos * dy_pos);
  }
}

void MovementSystem::apply_boundary(float &x, float &y) const {
  if (has_walls_) {
    x = std::clamp(x, 0.0f, world_width_);
    y = std::clamp(y, 0.0f, world_height_);
  } else {
    while (x < 0.0f)
      x += world_width_;
    while (x >= world_width_)
      x -= world_width_;
    while (y < 0.0f)
      y += world_height_;
    while (y >= world_height_)
      y -= world_height_;
  }
}

Vec2 MovementSystem::wrap_diff(Vec2 diff) const {
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

} // namespace moonai
