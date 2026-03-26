#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace moonai {

class Random {
public:
  explicit Random(std::uint64_t seed);

  int next_int(int min, int max);
  float next_float(float min, float max);
  float next_gaussian(float mean, float stddev);
  bool next_bool(float probability);

  int weighted_select(const std::vector<float> &weights);

  template <typename T> void shuffle(std::vector<T> &vec) {
    std::shuffle(vec.begin(), vec.end(), engine_);
  }

  std::vector<int> sample_indices(int total, int count);

  std::uint64_t seed() const {
    return seed_;
  }
  std::mt19937_64 &engine() {
    return engine_;
  }

private:
  std::uint64_t seed_;
  std::mt19937_64 engine_;
};

} // namespace moonai
