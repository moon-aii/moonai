#pragma once
#include <cstddef>
#include <type_traits>

namespace moonai {

template <typename T> struct ComponentTraits {
  static constexpr bool is_valid =
      std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;
  static constexpr bool gpu_aligned = false;
  static constexpr std::size_t max_count = 100000;
};

} // namespace moonai