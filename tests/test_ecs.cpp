#include "simulation/components.hpp"
#include "simulation/entity.hpp"
#include "simulation/registry.hpp"
#include "simulation/sparse_set.hpp"
#include <gtest/gtest.h>

using namespace moonai;

TEST(SparseSetTest, InsertAndRetrieve) {
  SparseSet set;
  Entity e1{1, 1};

  size_t idx = set.insert(e1);
  EXPECT_EQ(idx, 0);
  EXPECT_EQ(set.size(), 1);
  EXPECT_TRUE(set.contains(e1));

  Entity retrieved = set.get_entity(idx);
  EXPECT_EQ(retrieved, e1);
}

TEST(SparseSetTest, MultipleInsertions) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};
  Entity e3{3, 1};

  size_t idx1 = set.insert(e1);
  size_t idx2 = set.insert(e2);
  size_t idx3 = set.insert(e3);

  EXPECT_EQ(set.size(), 3);
  EXPECT_EQ(idx1, 0);
  EXPECT_EQ(idx2, 1);
  EXPECT_EQ(idx3, 2);

  EXPECT_TRUE(set.contains(e1));
  EXPECT_TRUE(set.contains(e2));
  EXPECT_TRUE(set.contains(e3));
}

TEST(SparseSetTest, DuplicateInsert) {
  SparseSet set;
  Entity e1{1, 1};

  size_t idx1 = set.insert(e1);
  size_t idx2 = set.insert(e1);

  EXPECT_EQ(idx1, idx2);
  EXPECT_EQ(set.size(), 1);
}

TEST(SparseSetTest, GetIndex) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};

  set.insert(e1);
  set.insert(e2);

  EXPECT_EQ(set.get_index(e1), 0);
  EXPECT_EQ(set.get_index(e2), 1);
}

TEST(SparseSetTest, GetIndexNotFound) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};

  set.insert(e1);

  size_t invalid = std::numeric_limits<uint32_t>::max();
  EXPECT_EQ(set.get_index(e2), invalid);
}

TEST(SparseSetTest, RemoveSingleEntity) {
  SparseSet set;
  Entity e1{1, 1};

  set.insert(e1);
  set.remove(e1);

  EXPECT_EQ(set.size(), 0);
  EXPECT_FALSE(set.contains(e1));
}

TEST(SparseSetTest, RemovePreservesOthers) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};
  Entity e3{3, 1};

  set.insert(e1);
  set.insert(e2);
  set.insert(e3);

  set.remove(e2);

  EXPECT_EQ(set.size(), 2);
  EXPECT_TRUE(set.contains(e1));
  EXPECT_FALSE(set.contains(e2));
  EXPECT_TRUE(set.contains(e3));
}

TEST(SparseSetTest, RemoveMiddleEntity) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};
  Entity e3{3, 1};

  set.insert(e1);
  set.insert(e2);
  set.insert(e3);

  set.remove(e2);

  EXPECT_EQ(set.get_index(e3), 1);
  EXPECT_EQ(set.get_entity(1), e3);
}

TEST(SparseSetTest, RemoveNonexistentEntity) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};

  set.insert(e1);
  set.remove(e2);

  EXPECT_EQ(set.size(), 1);
  EXPECT_TRUE(set.contains(e1));
}

TEST(SparseSetTest, DenseIteration) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e2{2, 1};
  Entity e3{3, 1};

  set.insert(e1);
  set.insert(e2);
  set.insert(e3);

  const auto &dense = set.dense();
  EXPECT_EQ(dense.size(), 3);

  bool has_e1 = false, has_e2 = false, has_e3 = false;
  for (const auto &e : dense) {
    if (e == e1)
      has_e1 = true;
    if (e == e2)
      has_e2 = true;
    if (e == e3)
      has_e3 = true;
  }

  EXPECT_TRUE(has_e1);
  EXPECT_TRUE(has_e2);
  EXPECT_TRUE(has_e3);
}

TEST(SparseSetTest, GenerationMismatch) {
  SparseSet set;
  Entity e1{1, 1};
  Entity e1_new_gen{1, 2};

  set.insert(e1);

  EXPECT_TRUE(set.contains(e1));
  EXPECT_FALSE(set.contains(e1_new_gen));
}
TEST(ECSRegistryTest, EntityCreation) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();

  EXPECT_NE(e1, e2);
  EXPECT_TRUE(registry.alive(e1));
  EXPECT_TRUE(registry.alive(e2));
  EXPECT_EQ(registry.size(), 2);
}

TEST(ECSRegistryTest, EntityDestruction) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();

  registry.destroy(e1);

  EXPECT_FALSE(registry.alive(e1));
  EXPECT_TRUE(registry.alive(e2));
  EXPECT_EQ(registry.size(), 1);
}

TEST(ECSRegistryTest, DestroyInvalidEntity) {
  Registry registry;

  registry.destroy(INVALID_ENTITY);

  auto e = registry.create();
  registry.destroy(e);
  registry.destroy(e);

  EXPECT_FALSE(registry.alive(e));
}

TEST(ECSRegistryTest, SlotRecycling) {
  Registry registry;

  auto e1 = registry.create();
  uint32_t original_index = e1.index;

  registry.destroy(e1);

  auto e2 = registry.create();

  EXPECT_EQ(e2.index, original_index);
  EXPECT_GT(e2.generation, e1.generation);
  EXPECT_NE(e1, e2);
}

TEST(ECSRegistryTest, ComponentArraysResize) {
  Registry registry;

  auto e = registry.create();
  size_t idx = registry.index_of(e);

  EXPECT_GT(registry.positions().size(), 0);
  EXPECT_GT(registry.vitals().size(), 0);
  EXPECT_GT(registry.identity().size(), 0);
}

TEST(ECSRegistryTest, DirectComponentAccess) {
  Registry registry;

  auto e = registry.create();

  registry.pos_x(e) = 100.0f;
  registry.pos_y(e) = 200.0f;
  registry.energy(e) = 50.0f;

  EXPECT_FLOAT_EQ(registry.pos_x(e), 100.0f);
  EXPECT_FLOAT_EQ(registry.pos_y(e), 200.0f);
  EXPECT_FLOAT_EQ(registry.energy(e), 50.0f);
}

TEST(ECSRegistryTest, LivingEntitiesList) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();
  auto e3 = registry.create();

  registry.destroy(e2);

  const auto &living = registry.living_entities();
  EXPECT_EQ(living.size(), 2);

  bool has_e1 = false, has_e3 = false;
  for (const auto &e : living) {
    if (e == e1)
      has_e1 = true;
    if (e == e3)
      has_e3 = true;
  }
  EXPECT_TRUE(has_e1);
  EXPECT_TRUE(has_e3);
}

TEST(ECSRegistryTest, Clear) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();

  registry.clear();

  EXPECT_EQ(registry.size(), 0);
  EXPECT_FALSE(registry.alive(e1));
  EXPECT_FALSE(registry.alive(e2));
  EXPECT_TRUE(registry.empty());
}

TEST(ECSRegistryTest, SoAComponentsAccess) {
  Registry registry;

  auto e1 = registry.create();
  auto e2 = registry.create();

  size_t idx1 = registry.index_of(e1);
  size_t idx2 = registry.index_of(e2);

  registry.positions().x[idx1] = 10.0f;
  registry.positions().y[idx1] = 20.0f;
  registry.positions().x[idx2] = 30.0f;
  registry.positions().y[idx2] = 40.0f;

  EXPECT_FLOAT_EQ(registry.positions().x[idx1], 10.0f);
  EXPECT_FLOAT_EQ(registry.positions().y[idx1], 20.0f);
  EXPECT_FLOAT_EQ(registry.positions().x[idx2], 30.0f);
  EXPECT_FLOAT_EQ(registry.positions().y[idx2], 40.0f);
}

TEST(ECSRegistryTest, VitalsSoA) {
  Registry registry;

  auto e = registry.create();
  size_t idx = registry.index_of(e);

  registry.vitals().energy[idx] = 100.0f;
  registry.vitals().age[idx] = 5;
  registry.vitals().alive[idx] = 1;
  registry.vitals().reproduction_cooldown[idx] = 10;

  EXPECT_FLOAT_EQ(registry.vitals().energy[idx], 100.0f);
  EXPECT_EQ(registry.vitals().age[idx], 5);
  EXPECT_EQ(registry.vitals().alive[idx], 1);
  EXPECT_EQ(registry.vitals().reproduction_cooldown[idx], 10);
}

TEST(ECSRegistryTest, IdentitySoA) {
  Registry registry;

  auto e = registry.create();
  size_t idx = registry.index_of(e);

  registry.identity().type[idx] = IdentitySoA::TYPE_PREDATOR;
  registry.identity().species_id[idx] = 42;
  registry.identity().entity_id[idx] = 123;

  EXPECT_EQ(registry.identity().type[idx], IdentitySoA::TYPE_PREDATOR);
  EXPECT_EQ(registry.identity().species_id[idx], 42);
  EXPECT_EQ(registry.identity().entity_id[idx], 123);
}

TEST(ECSRegistryTest, SensorSoA) {
  Registry registry;

  auto e = registry.create();
  size_t idx = registry.index_of(e);

  float *inputs = registry.sensors().input_ptr(idx);
  for (int i = 0; i < SensorSoA::INPUT_COUNT; ++i) {
    inputs[i] = static_cast<float>(i);
  }

  for (int i = 0; i < SensorSoA::INPUT_COUNT; ++i) {
    EXPECT_FLOAT_EQ(registry.sensors().inputs[idx * SensorSoA::INPUT_COUNT + i],
                    static_cast<float>(i));
  }

  float *outputs = registry.sensors().output_ptr(idx);
  outputs[0] = 0.5f;
  outputs[1] = -0.5f;

  EXPECT_FLOAT_EQ(registry.sensors().outputs[idx * SensorSoA::OUTPUT_COUNT + 0],
                  0.5f);
  EXPECT_FLOAT_EQ(registry.sensors().outputs[idx * SensorSoA::OUTPUT_COUNT + 1],
                  -0.5f);
}