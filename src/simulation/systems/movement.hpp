#pragma once
#include "core/types.hpp"
#include "simulation/system.hpp"

namespace moonai {

class MovementSystem : public System {
public:
  MovementSystem(float world_width, float world_height, bool has_walls);

  void update(Registry &registry) override;
  const char *name() const override {
    return "MovementSystem";
  }

private:
  float world_width_;
  float world_height_;
  bool has_walls_;

  void apply_boundary(float &x, float &y) const;
  Vec2 wrap_diff(Vec2 diff) const;
};

} // namespace moonai
