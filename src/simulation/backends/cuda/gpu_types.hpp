#pragma once

namespace moonai::gpu {

struct alignas(16) GpuPopulationEntry {
  unsigned int id;
  float pos_x;
  float pos_y;
  float padding;
};

struct alignas(16) GpuFoodEntry {
  unsigned int id;
  float pos_x;
  float pos_y;
  float padding;
};

} // namespace moonai::gpu
