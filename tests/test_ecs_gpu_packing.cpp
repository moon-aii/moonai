#include "gpu/ecs_gpu_packing.hpp"
#include "gpu/gpu_data_buffer.hpp"
#include "gpu/gpu_entity_mapping.hpp"
#include "simulation/components.hpp"
#include "simulation/entity.hpp"
#include "simulation/registry.hpp"
#include <gtest/gtest.h>

using namespace moonai;
using namespace moonai::gpu;

class EcsGpuPackingTest : public ::testing::Test {
protected:
  void SetUp() override {
    for (int i = 0; i < 10; ++i) {
      Entity e = registry_.create();
      size_t idx = registry_.index_of(e);

      registry_.positions().x[idx] = static_cast<float>(i * 10.0f);
      registry_.positions().y[idx] = static_cast<float>(i * 5.0f);
      registry_.motion().vel_x[idx] = static_cast<float>(i * 0.5f);
      registry_.motion().vel_y[idx] = static_cast<float>(i * 0.3f);
      registry_.vitals().energy[idx] = static_cast<float>(50.0f + i * 5.0f);
      registry_.vitals().age[idx] = i;
      registry_.vitals().alive[idx] = (i % 3 != 0);
      registry_.identity().type[idx] =
          (i % 2 == 0) ? IdentitySoA::TYPE_PREDATOR : IdentitySoA::TYPE_PREY;
      registry_.identity().species_id[idx] = i / 2;

      entities_.push_back(e);
    }
  }

  Registry registry_;
  std::vector<Entity> entities_;
};

TEST(GpuEntityMappingTest, BasicMapping) {
  GpuEntityMapping mapping;
  mapping.resize(100);

  // Create some test entities
  std::vector<Entity> living;
  for (int i = 0; i < 5; ++i) {
    living.push_back(Entity{static_cast<uint32_t>(i + 1), 1});
  }

  mapping.build(living);

  EXPECT_EQ(mapping.count(), 5u);
  EXPECT_FALSE(mapping.empty());

  for (uint32_t i = 0; i < 5; ++i) {
    Entity e{static_cast<uint32_t>(i + 1), 1};
    EXPECT_EQ(mapping.gpu_index(e), static_cast<int32_t>(i));
  }

  for (uint32_t i = 0; i < 5; ++i) {
    Entity e = mapping.entity_at(i);
    EXPECT_EQ(e.index, i + 1);
    EXPECT_EQ(e.generation, 1u);
  }
}

TEST(GpuEntityMappingTest, NonContiguousMapping) {
  GpuEntityMapping mapping;
  mapping.resize(100);

  // Create entities with gaps
  std::vector<Entity> living;
  living.push_back(Entity{5, 1});
  living.push_back(Entity{10, 1});
  living.push_back(Entity{20, 1});

  mapping.build(living);

  EXPECT_EQ(mapping.count(), 3u);

  EXPECT_EQ(mapping.gpu_index(Entity{5, 1}), 0);
  EXPECT_EQ(mapping.gpu_index(Entity{10, 1}), 1);
  EXPECT_EQ(mapping.gpu_index(Entity{20, 1}), 2);

  EXPECT_EQ(mapping.gpu_index(Entity{1, 1}), -1);
  EXPECT_EQ(mapping.gpu_index(Entity{15, 1}), -1);
}

TEST(GpuEntityMappingTest, ClearMapping) {
  GpuEntityMapping mapping;
  mapping.resize(100);

  std::vector<Entity> living = {Entity{1, 1}, Entity{2, 1}, Entity{3, 1}};
  mapping.build(living);

  EXPECT_EQ(mapping.count(), 3u);

  mapping.clear();

  EXPECT_EQ(mapping.count(), 0u);
  EXPECT_TRUE(mapping.empty());
  EXPECT_EQ(mapping.gpu_index(Entity{1, 1}), -1);
}

TEST_F(EcsGpuPackingTest, PackEcsToGpu) {
  GpuEntityMapping mapping;
  mapping.resize(100);

  std::vector<Entity> living = registry_.living_entities();
  mapping.build(living);

  EXPECT_EQ(mapping.count(), 10u);

  GpuDataBuffer buffer(100);
  pack_ecs_to_gpu(registry_, mapping, buffer);

  for (uint32_t gpu_idx = 0; gpu_idx < mapping.count(); ++gpu_idx) {
    Entity entity = mapping.entity_at(gpu_idx);
    size_t ecs_idx = registry_.index_of(entity);

    EXPECT_FLOAT_EQ(buffer.host_positions_x()[gpu_idx],
                    registry_.positions().x[ecs_idx]);
    EXPECT_FLOAT_EQ(buffer.host_positions_y()[gpu_idx],
                    registry_.positions().y[ecs_idx]);

    EXPECT_FLOAT_EQ(buffer.host_velocities_x()[gpu_idx],
                    registry_.motion().vel_x[ecs_idx]);
    EXPECT_FLOAT_EQ(buffer.host_velocities_y()[gpu_idx],
                    registry_.motion().vel_y[ecs_idx]);

    EXPECT_FLOAT_EQ(buffer.host_energy()[gpu_idx],
                    registry_.vitals().energy[ecs_idx]);

    EXPECT_EQ(buffer.host_alive()[gpu_idx], registry_.vitals().alive[ecs_idx]);

    EXPECT_EQ(buffer.host_types()[gpu_idx],
              static_cast<uint8_t>(registry_.identity().type[ecs_idx]));
  }
}

TEST_F(EcsGpuPackingTest, UnpackGpuToEcs) {
  GpuEntityMapping mapping;
  mapping.resize(100);

  std::vector<Entity> living = registry_.living_entities();
  mapping.build(living);

  GpuDataBuffer buffer(100);

  for (uint32_t gpu_idx = 0; gpu_idx < mapping.count(); ++gpu_idx) {
    buffer.host_outputs_energy()[gpu_idx] = 42.0f;
    buffer.host_outputs_alive()[gpu_idx] = 1;
    buffer.host_outputs_velocities_x()[gpu_idx] = 1.5f;
    buffer.host_outputs_velocities_y()[gpu_idx] = -2.5f;
    buffer.host_positions_x()[gpu_idx] = 100.0f + gpu_idx;
    buffer.host_positions_y()[gpu_idx] = 200.0f + gpu_idx;
  }

  unpack_gpu_to_ecs(buffer, mapping, registry_);

  for (uint32_t gpu_idx = 0; gpu_idx < mapping.count(); ++gpu_idx) {
    Entity entity = mapping.entity_at(gpu_idx);
    size_t ecs_idx = registry_.index_of(entity);

    EXPECT_FLOAT_EQ(registry_.vitals().energy[ecs_idx], 42.0f);
    EXPECT_EQ(registry_.vitals().alive[ecs_idx], 1);
    EXPECT_FLOAT_EQ(registry_.motion().vel_x[ecs_idx], 1.5f);
    EXPECT_FLOAT_EQ(registry_.motion().vel_y[ecs_idx], -2.5f);
    EXPECT_FLOAT_EQ(registry_.positions().x[ecs_idx], 100.0f + gpu_idx);
    EXPECT_FLOAT_EQ(registry_.positions().y[ecs_idx], 200.0f + gpu_idx);
  }
}

TEST(GpuEntityMappingTest, EmptyMapping) {
  GpuEntityMapping mapping;
  mapping.resize(100);

  std::vector<Entity> empty;
  mapping.build(empty);

  EXPECT_EQ(mapping.count(), 0u);
  EXPECT_TRUE(mapping.empty());
  EXPECT_EQ(mapping.gpu_index(Entity{1, 1}), -1);
  EXPECT_EQ(mapping.entity_at(0).index, INVALID_ENTITY.index);
}

TEST_F(EcsGpuPackingTest, PrepareEcsForGpu) {
  GpuEntityMapping mapping;
  GpuDataBuffer buffer(100);

  uint32_t count = prepare_ecs_for_gpu(registry_, mapping, buffer);

  EXPECT_EQ(count, 10u);
  EXPECT_EQ(mapping.count(), 10u);

  EXPECT_FLOAT_EQ(buffer.host_positions_x()[0], 0.0f);
  EXPECT_FLOAT_EQ(buffer.host_positions_y()[0], 0.0f);
  EXPECT_FLOAT_EQ(buffer.host_positions_x()[1], 10.0f);
  EXPECT_FLOAT_EQ(buffer.host_positions_y()[1], 5.0f);
}

TEST_F(EcsGpuPackingTest, ApplyGpuResults) {
  GpuEntityMapping mapping;
  GpuDataBuffer buffer(100);

  prepare_ecs_for_gpu(registry_, mapping, buffer);

  for (uint32_t i = 0; i < mapping.count(); ++i) {
    buffer.host_outputs_energy()[i] = 99.0f;
    buffer.host_positions_x()[i] = 999.0f;
    buffer.host_positions_y()[i] = 888.0f;
  }

  apply_gpu_results(buffer, mapping, registry_);

  for (uint32_t gpu_idx = 0; gpu_idx < mapping.count(); ++gpu_idx) {
    Entity entity = mapping.entity_at(gpu_idx);
    size_t ecs_idx = registry_.index_of(entity);

    EXPECT_FLOAT_EQ(registry_.vitals().energy[ecs_idx], 99.0f);
    EXPECT_FLOAT_EQ(registry_.positions().x[ecs_idx], 999.0f);
    EXPECT_FLOAT_EQ(registry_.positions().y[ecs_idx], 888.0f);
  }
}
