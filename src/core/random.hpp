#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace moonai {

class Random {
public:
  explicit Random(int seed);

  int next_int(int min, int max);
  float next_float(float min, float max);
  float next_gaussian(float mean, float stddev);
  bool next_bool(float probability);

  int weighted_select(const std::vector<float> &weights);

  std::vector<int> sample_indices(int total, int count);

  int seed() const {
    return seed_;
  }

private:
  int seed_;
  std::mt19937_64 engine_;
};

} // namespace moonai
