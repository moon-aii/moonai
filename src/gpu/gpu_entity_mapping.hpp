#pragma once
#include "simulation/entity.hpp"
#include <cstdint>
#include <vector>

namespace moonai {
namespace gpu {

class GpuEntityMapping {
public:
  GpuEntityMapping() = default;

  void resize(std::size_t max_entities);

  void build(const std::vector<Entity> &living);
  void build_count(std::size_t count);

  [[nodiscard]] int32_t gpu_index(Entity e) const noexcept {
    if (e.index >= entity_to_gpu_.size()) {
      return -1;
    }
    return entity_to_gpu_[e.index];
  }

  [[nodiscard]] Entity entity_at(uint32_t gpu_idx) const noexcept {
    if (gpu_idx >= gpu_to_entity_.size()) {
      return INVALID_ENTITY;
    }
    return gpu_to_entity_[gpu_idx];
  }

  [[nodiscard]] uint32_t count() const noexcept {
    return count_;
  }

  [[nodiscard]] bool empty() const noexcept {
    return count_ == 0;
  }

  [[nodiscard]] const std::vector<int32_t> &entity_to_gpu() const noexcept {
    return entity_to_gpu_;
  }

  [[nodiscard]] const std::vector<Entity> &gpu_to_entity() const noexcept {
    return gpu_to_entity_;
  }

  void clear() noexcept {
    // Reset entity-to-GPU to -1 (we don't clear the vector to preserve
    // capacity)
    std::fill(entity_to_gpu_.begin(), entity_to_gpu_.end(), -1);
    count_ = 0;
  }

private:
  std::vector<int32_t> entity_to_gpu_;

  std::vector<Entity> gpu_to_entity_;

  uint32_t count_ = 0;
};

} // namespace gpu
} // namespace moonai
