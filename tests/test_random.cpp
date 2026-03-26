#include "core/random.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <set>

using namespace moonai;

TEST(RandomTest, Deterministic) {
  Random rng1(42);
  Random rng2(42);

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(rng1.next_int(0, 1000), rng2.next_int(0, 1000));
  }
}

TEST(RandomTest, DeterministicFloat) {
  Random rng1(123);
  Random rng2(123);

  for (int i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(rng1.next_float(-1.0f, 1.0f), rng2.next_float(-1.0f, 1.0f));
  }
}

TEST(RandomTest, DifferentSeedsDifferentSequences) {
  Random rng1(1);
  Random rng2(2);

  bool any_different = false;
  for (int i = 0; i < 10; ++i) {
    if (rng1.next_int(0, 1000000) != rng2.next_int(0, 1000000)) {
      any_different = true;
      break;
    }
  }
  EXPECT_TRUE(any_different);
}

TEST(RandomTest, IntRange) {
  Random rng(42);
  for (int i = 0; i < 1000; ++i) {
    int val = rng.next_int(5, 10);
    EXPECT_GE(val, 5);
    EXPECT_LE(val, 10);
  }
}

TEST(RandomTest, FloatRange) {
  Random rng(42);
  for (int i = 0; i < 1000; ++i) {
    float val = rng.next_float(-1.0f, 1.0f);
    EXPECT_GE(val, -1.0f);
    EXPECT_LT(val, 1.0f);
  }
}

TEST(RandomTest, BoolProbability) {
  Random rng(42);
  int true_count = 0;
  int total = 10000;
  for (int i = 0; i < total; ++i) {
    if (rng.next_bool(0.3f))
      ++true_count;
  }
  // Should be roughly 30% +/- 3%
  float ratio = static_cast<float>(true_count) / total;
  EXPECT_NEAR(ratio, 0.3f, 0.03f);
}

TEST(RandomTest, WeightedSelectRespectsWeights) {
  Random rng(42);
  std::vector<float> weights = {0.0f, 0.0f, 1.0f, 0.0f};

  // All weight on index 2
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(rng.weighted_select(weights), 2);
  }
}

TEST(RandomTest, WeightedSelectDistribution) {
  Random rng(42);
  std::vector<float> weights = {1.0f, 3.0f}; // 25% vs 75%
  int counts[2] = {0, 0};
  int total = 10000;

  for (int i = 0; i < total; ++i) {
    counts[rng.weighted_select(weights)]++;
  }

  float ratio = static_cast<float>(counts[1]) / total;
  EXPECT_NEAR(ratio, 0.75f, 0.03f);
}

TEST(RandomTest, WeightedSelectEmptyWeights) {
  Random rng(42);
  std::vector<float> weights;
  EXPECT_EQ(rng.weighted_select(weights), -1);
}

TEST(RandomTest, ShuffleMaintainsElements) {
  Random rng(42);
  std::vector<int> original = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> shuffled = original;
  rng.shuffle(shuffled);

  std::set<int> orig_set(original.begin(), original.end());
  std::set<int> shuf_set(shuffled.begin(), shuffled.end());
  EXPECT_EQ(orig_set, shuf_set);
}

TEST(RandomTest, ShuffleIsDeterministic) {
  std::vector<int> v1 = {1, 2, 3, 4, 5};
  std::vector<int> v2 = {1, 2, 3, 4, 5};

  Random rng1(99);
  Random rng2(99);
  rng1.shuffle(v1);
  rng2.shuffle(v2);

  EXPECT_EQ(v1, v2);
}

TEST(RandomTest, SampleIndices) {
  Random rng(42);
  auto indices = rng.sample_indices(100, 10);

  EXPECT_EQ(indices.size(), 10u);

  std::set<int> unique(indices.begin(), indices.end());
  EXPECT_EQ(unique.size(), 10u);

  for (int idx : indices) {
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, 100);
  }
}

TEST(RandomTest, GaussianReasonableRange) {
  Random rng(42);
  float sum = 0.0f;
  int n = 10000;
  for (int i = 0; i < n; ++i) {
    sum += rng.next_gaussian(5.0f, 1.0f);
  }
  float mean = sum / n;
  EXPECT_NEAR(mean, 5.0f, 0.1f);
}
