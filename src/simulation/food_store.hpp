#pragma once

#include "core/config.hpp"
#include "core/random.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moonai {

class FoodStore {
public:
  void initialize(const SimulationConfig &config, Random &rng);
  void respawn_step(const SimulationConfig &config, int step_index,
                    std::uint64_t seed);

  std::size_t size() const {
    return pos_x_.size();
  }

  const std::vector<float> &pos_x() const {
    return pos_x_;
  }
  const std::vector<float> &pos_y() const {
    return pos_y_;
  }
  const std::vector<uint8_t> &active() const {
    return active_;
  }
  const std::vector<uint32_t> &slot_index() const {
    return slot_index_;
  }

  std::vector<float> &pos_x() {
    return pos_x_;
  }
  std::vector<float> &pos_y() {
    return pos_y_;
  }
  std::vector<uint8_t> &active() {
    return active_;
  }

private:
  std::vector<float> pos_x_;
  std::vector<float> pos_y_;
  std::vector<uint8_t> active_;
  std::vector<uint32_t> slot_index_;
};

} // namespace moonai
