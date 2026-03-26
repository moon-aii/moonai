#include "core/random.hpp"
#include <gtest/gtest.h>

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

TEST(RandomTest, WeightedSelectRespectsWeights) {
  Random rng(42);
  std::vector<float> weights = {0.0f, 0.0f, 1.0f, 0.0f};

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(rng.weighted_select(weights), 2);
  }
}

TEST(RandomTest, WeightedSelectEmptyWeights) {
  Random rng(42);
  std::vector<float> weights;
  EXPECT_EQ(rng.weighted_select(weights), -1);
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
