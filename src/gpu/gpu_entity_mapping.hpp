#pragma once
#include "simulation/entity.hpp"
#include <cstdint>
#include <vector>

namespace moonai {
namespace gpu {

/**
 * @brief Mapping between Entity handles and contiguous GPU buffer indices
 *
 * ECS uses sparse-set storage where Entity handles are stable but component
 * arrays have gaps (dead entities). GPU kernels need contiguous data.
 *
 * This class manages the bidirectional mapping:
 * - Entity → GPU index (for packing ECS data into GPU buffers)
 * - GPU index → Entity (for unpacking results back to ECS)
 *
 * The mapping is rebuilt each frame from the list of living entities.
 * This is O(N) where N is the number of living entities.
 */
class GpuEntityMapping {
public:
  GpuEntityMapping() = default;

  /**
   * @brief Resize internal storage for maximum entity capacity
   * @param max_entities Maximum number of entities that might be mapped
   */
  void resize(std::size_t max_entities);

  /**
   * @brief Build mapping from list of living entities
   * @param living Vector of living entity handles
   *
   * This packs all living entities into contiguous GPU indices [0, count).
   * The mapping is built fresh each frame.
   */
  void build(const std::vector<Entity> &living);

  /**
   * @brief Get GPU buffer index for an entity
   * @param e Entity handle
   * @return GPU index (0 to count-1) or -1 if entity not in mapping
   */
  [[nodiscard]] int32_t gpu_index(Entity e) const noexcept {
    if (e.index >= entity_to_gpu_.size()) {
      return -1;
    }
    return entity_to_gpu_[e.index];
  }

  /**
   * @brief Get entity handle at a GPU buffer index
   * @param gpu_idx GPU buffer index
   * @return Entity handle or INVALID_ENTITY if out of range
   */
  [[nodiscard]] Entity entity_at(uint32_t gpu_idx) const noexcept {
    if (gpu_idx >= gpu_to_entity_.size()) {
      return INVALID_ENTITY;
    }
    return gpu_to_entity_[gpu_idx];
  }

  /** @return Number of entities in the mapping */
  [[nodiscard]] uint32_t count() const noexcept {
    return count_;
  }

  /** @return True if mapping is empty */
  [[nodiscard]] bool empty() const noexcept {
    return count_ == 0;
  }

  /** @return Raw access to entity-to-GPU mapping (for GPU packing) */
  [[nodiscard]] const std::vector<int32_t> &entity_to_gpu() const noexcept {
    return entity_to_gpu_;
  }

  /** @return Raw access to GPU-to-entity mapping (for GPU unpacking) */
  [[nodiscard]] const std::vector<Entity> &gpu_to_entity() const noexcept {
    return gpu_to_entity_;
  }

  /**
   * @brief Clear the mapping (reset count, but keep capacity)
   */
  void clear() noexcept {
    // Reset entity-to-GPU to -1 (we don't clear the vector to preserve
    // capacity)
    std::fill(entity_to_gpu_.begin(), entity_to_gpu_.end(), -1);
    count_ = 0;
  }

private:
  // Entity index → GPU index (or -1 if not mapped)
  // Size = max_entities, sparse mapping
  std::vector<int32_t> entity_to_gpu_;

  // GPU index → Entity
  // Size = count, dense contiguous mapping
  std::vector<Entity> gpu_to_entity_;

  // Number of entities currently mapped
  uint32_t count_ = 0;
};

} // namespace gpu
} // namespace moonai
