#pragma once

#include "core/config.hpp"
#include "core/random.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moonai {

struct FoodStore {
  void initialize(const SimulationConfig &config, Random &rng);
  void respawn_step(const SimulationConfig &config, int step_index,
                    std::uint64_t seed);

  std::vector<float> pos_x;
  std::vector<float> pos_y;
  std::vector<uint8_t> active;

  std::size_t size() const {
    return pos_x.size();
  }

private:
  std::vector<uint32_t> slot_index;
};

} // namespace moonai
