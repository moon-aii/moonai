#pragma once

#include <cmath>
#include <cstdint>

namespace moonai {

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;

  Vec2 operator-(const Vec2 &other) const {
    return {x - other.x, y - other.y};
  }

  float length() const {
    return std::sqrt(x * x + y * y);
  }
};

} // namespace moonai
