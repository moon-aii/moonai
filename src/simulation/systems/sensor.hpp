#pragma once
#include "core/types.hpp"
#include "simulation/spatial_grid_ecs.hpp"
#include "simulation/system.hpp"

namespace moonai {

class SensorSystem : public System {
public:
  SensorSystem(const SpatialGridECS &agent_grid, float world_width,
               float world_height, float max_energy, bool has_walls);

  void update(Registry &registry, float dt) override;
  const char *name() const override {
    return "SensorSystem";
  }

private:
  const SpatialGridECS &agent_grid_;

  float world_width_;
  float world_height_;
  float max_energy_;
  bool has_walls_;

  static constexpr float VISION_RANGE = 200.0f; // Default vision range
  static constexpr float MAX_DENSITY = 10.0f;   // For density normalization

  void build_sensors_for_entity(size_t entity_idx, Registry &registry);
  Vec2 wrap_diff(Vec2 diff) const;
  float normalize_angle(float dx, float dy) const;
};

} // namespace moonai