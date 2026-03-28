#pragma once
#include "core/types.hpp"
#include "simulation/entity.hpp"
#include <unordered_map>
#include <vector>

namespace moonai {

class SpatialGrid {
public:
  SpatialGrid(float world_width, float world_height, float cell_size);

  void clear();
  void insert(Entity e, Vec2 pos);

  // Query entities within radius of position
  std::vector<Entity> query_radius(Vec2 center, float radius) const;

  // Get all entities in grid
  const std::vector<Entity> &all_entities() const {
    return entities_;
  }

  float cell_size() const {
    return cell_size_;
  }

private:
  using CellKey = uint64_t;

  float cell_size_;
  float world_width_;
  float world_height_;
  int cols_;
  int rows_;

  std::unordered_map<CellKey, std::vector<Entity>> cells_;
  std::vector<Entity> entities_;

  CellKey cell_key(int x, int y) const {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
           static_cast<uint32_t>(y);
  }

  CellKey cell_key(Vec2 pos) const;
  int cell_x(float x) const;
  int cell_y(float y) const;
};

} // namespace moonai
