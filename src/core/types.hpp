#pragma once

#include <cmath>
#include <cstdint>

namespace moonai {

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;

  Vec2 operator+(const Vec2 &other) const {
    return {x + other.x, y + other.y};
  }
  Vec2 operator-(const Vec2 &other) const {
    return {x - other.x, y - other.y};
  }
  Vec2 operator*(float scalar) const {
    return {x * scalar, y * scalar};
  }

  float length() const {
    return std::sqrt(x * x + y * y);
  }

  Vec2 normalized() const {
    float len = length();
    if (len < 1e-6f)
      return {0.0f, 0.0f};
    return {x / len, y / len};
  }

  float distance_to(const Vec2 &other) const {
    return (*this - other).length();
  }
};

} // namespace moonai
