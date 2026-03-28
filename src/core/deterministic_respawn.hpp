#pragma once

#include <cstdint>

namespace moonai::respawn {

#ifdef __CUDACC__
#define MOONAI_HD __host__ __device__
#else
#define MOONAI_HD
#endif

MOONAI_HD inline std::uint32_t hash_u32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

MOONAI_HD inline std::uint32_t base_seed(std::uint64_t seed, int step_index,
                                         std::uint32_t item_id) {
  return static_cast<std::uint32_t>(seed) ^
         (static_cast<std::uint32_t>(step_index + 1) * 0x9e3779b9U) ^
         (item_id * 0x85ebca6bU);
}

MOONAI_HD inline float unit_float(std::uint32_t seed) {
  return static_cast<float>(hash_u32(seed)) / static_cast<float>(0xffffffffU);
}

MOONAI_HD inline bool should_respawn(std::uint64_t seed, int step_index,
                                     std::uint32_t item_id,
                                     float respawn_rate) {
  return unit_float(base_seed(seed, step_index, item_id)) < respawn_rate;
}

MOONAI_HD inline float respawn_x(std::uint64_t seed, int step_index,
                                 std::uint32_t item_id, float world_width) {
  return unit_float(base_seed(seed, step_index, item_id) ^ 0x68bc21ebU) *
         world_width;
}

MOONAI_HD inline float respawn_y(std::uint64_t seed, int step_index,
                                 std::uint32_t item_id, float world_height) {
  return unit_float(base_seed(seed, step_index, item_id) ^ 0x02e5be93U) *
         world_height;
}

#undef MOONAI_HD

} // namespace moonai::respawn
