#include "simulation/spatial_grid.hpp"
#include <algorithm>
#include <cmath>

namespace moonai {

SpatialGrid::SpatialGrid(float world_width, float world_height, float cell_size)
    : cell_size_(cell_size), world_width_(world_width),
      world_height_(world_height) {
  cols_ = static_cast<int>(std::ceil(world_width / cell_size));
  rows_ = static_cast<int>(std::ceil(world_height / cell_size));
}

void SpatialGrid::clear() {
  cells_.clear();
  entities_.clear();
}

void SpatialGrid::insert(Entity e, Vec2 pos) {
  CellKey key = cell_key(pos);
  cells_[key].push_back(e);
  entities_.push_back(e);
}

std::vector<Entity> SpatialGrid::query_radius(Vec2 center, float radius) const {
  std::vector<Entity> result;
  float radius_sq = radius * radius;

  // Calculate cell range to check
  int min_x = cell_x(center.x - radius);
  int max_x = cell_x(center.x + radius);
  int min_y = cell_y(center.y - radius);
  int max_y = cell_y(center.y + radius);

  // Clamp to valid range
  min_x = std::max(0, min_x);
  max_x = std::min(cols_ - 1, max_x);
  min_y = std::max(0, min_y);
  max_y = std::min(rows_ - 1, max_y);

  // Query cells in range
  for (int y = min_y; y <= max_y; ++y) {
    for (int x = min_x; x <= max_x; ++x) {
      CellKey key = cell_key(x, y);
      auto it = cells_.find(key);
      if (it != cells_.end()) {
        for (Entity e : it->second) {
          result.push_back(e);
        }
      }
    }
  }

  return result;
}

SpatialGrid::CellKey SpatialGrid::cell_key(Vec2 pos) const {
  return cell_key(cell_x(pos.x), cell_y(pos.y));
}

int SpatialGrid::cell_x(float x) const {
  return static_cast<int>(x / cell_size_);
}

int SpatialGrid::cell_y(float y) const {
  return static_cast<int>(y / cell_size_);
}

} // namespace moonai