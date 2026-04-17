#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace moonai {

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;

  float length() const {
    return std::sqrt(x * x + y * y);
  }
};

constexpr uint32_t INVALID_ENTITY = std::numeric_limits<uint32_t>::max();

// Agent neural network topology constants
inline constexpr int SENSOR_COUNT = 35;
inline constexpr int OUTPUT_COUNT = 2;

} // namespace moonai
