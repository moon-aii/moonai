#pragma once
#include <cstdint>
#include <functional>

namespace moonai {

struct Entity {
  uint32_t index;
  uint32_t generation;

  bool operator==(const Entity &other) const {
    return index == other.index && generation == other.generation;
  }
  bool operator!=(const Entity &other) const {
    return !(*this == other);
  }
  bool valid() const {
    return generation != 0;
  }
};

constexpr Entity INVALID_ENTITY = {0, 0};

// Hash function for unordered_map
struct EntityHash {
  size_t operator()(const Entity &e) const {
    return std::hash<uint64_t>{}((static_cast<uint64_t>(e.index) << 32) |
                                 e.generation);
  }
};

} // namespace moonai