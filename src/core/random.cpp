#include "core/random.hpp"

namespace moonai {

Random::Random(std::uint64_t seed) : seed_(seed), engine_(seed) {}

int Random::next_int(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(engine_);
}

float Random::next_float(float min, float max) {
  std::uniform_real_distribution<float> dist(min, max);
  return dist(engine_);
}

float Random::next_gaussian(float mean, float stddev) {
  std::normal_distribution<float> dist(mean, stddev);
  return dist(engine_);
}

bool Random::next_bool(float probability) {
  return next_float(0.0f, 1.0f) < probability;
}

int Random::weighted_select(const std::vector<float> &weights) {
  if (weights.empty())
    return -1;

  float total = std::accumulate(weights.begin(), weights.end(), 0.0f);
  if (total <= 0.0f)
    return next_int(0, static_cast<int>(weights.size()) - 1);

  float r = next_float(0.0f, total);
  float cumulative = 0.0f;
  for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
    cumulative += weights[i];
    if (r <= cumulative)
      return i;
  }
  return static_cast<int>(weights.size()) - 1;
}

std::vector<int> Random::sample_indices(int total, int count) {
  std::vector<int> indices(total);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), engine_);
  if (count < total) {
    indices.resize(count);
  }
  return indices;
}

} // namespace moonai
