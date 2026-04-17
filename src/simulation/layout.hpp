#pragma once

namespace moonai::simulation {

struct alignas(16) PopulationEntry {
  unsigned int id;
  float pos_x;
  float pos_y;
  float padding;
};

struct alignas(16) FoodEntry {
  unsigned int id;
  float pos_x;
  float pos_y;
  float padding;
};

} // namespace moonai::simulation
