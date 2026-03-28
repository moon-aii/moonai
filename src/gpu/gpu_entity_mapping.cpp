#include "gpu/gpu_entity_mapping.hpp"
#include <algorithm>

namespace moonai {
namespace gpu {

void GpuEntityMapping::resize(std::size_t max_entities) {
  entity_to_gpu_.resize(max_entities, -1);
  gpu_to_entity_.resize(max_entities);
}

void GpuEntityMapping::build(const std::vector<Entity> &living) {
  // Clear existing mapping
  clear();

  // Ensure we have enough space
  if (living.size() > gpu_to_entity_.size()) {
    gpu_to_entity_.resize(living.size());
  }

  // Build mapping: living entities get GPU indices [0, count)
  uint32_t gpu_idx = 0;
  for (const Entity &e : living) {
    // Update entity-to-GPU mapping
    if (e.index >= entity_to_gpu_.size()) {
      entity_to_gpu_.resize(e.index + 1, -1);
    }
    entity_to_gpu_[e.index] = static_cast<int32_t>(gpu_idx);

    // Update GPU-to-entity mapping
    gpu_to_entity_[gpu_idx] = e;

    ++gpu_idx;
  }

  count_ = gpu_idx;
}

void GpuEntityMapping::build_count(std::size_t count) {
  clear();
  if (count > gpu_to_entity_.size()) {
    gpu_to_entity_.resize(count);
  }
  count_ = static_cast<uint32_t>(count);
}

} // namespace gpu
} // namespace moonai
