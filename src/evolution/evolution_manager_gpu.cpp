#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "gpu/gpu_batch.hpp"
#include "gpu/gpu_network_cache.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace moonai {

void EvolutionManager::enable_gpu(bool use_gpu) {
  use_gpu_ = use_gpu;
  if (use_gpu_ && !gpu_network_cache_) {
    gpu_network_cache_ = std::make_unique<gpu::GpuNetworkCache>();
    gpu_network_cache_->invalidate();
    spdlog::info("GPU neural inference enabled");
  } else if (!use_gpu_) {
    gpu_network_cache_.reset();
    spdlog::info("GPU neural inference disabled");
  }
}

bool EvolutionManager::launch_gpu_neural(AppState &state,
                                         gpu::GpuBatch &gpu_batch,
                                         std::size_t agent_count) {
  MOONAI_PROFILE_SCOPE("gpu_neural", gpu_batch.stream());

  if (!gpu_network_cache_) {
    spdlog::error("GPU neural cache not initialized");
    return false;
  }

  std::vector<std::pair<uint32_t, int>> network_entities_with_indices;
  {
    MOONAI_PROFILE_SCOPE("gpu_network_scan");
    network_entities_with_indices.reserve(agent_count);

    const uint32_t entity_count = static_cast<uint32_t>(agent_count);
    for (uint32_t entity = 0; entity < entity_count; ++entity) {
      const int gpu_idx = static_cast<int>(entity);
      if (state.evolution.network_cache.has(entity)) {
        network_entities_with_indices.emplace_back(entity, gpu_idx);
      }
    }

    if (network_entities_with_indices.empty()) {
      spdlog::warn("No entities with neural networks found in GPU batch");
      return true;
    }
  }

  if (gpu_network_cache_->is_dirty() ||
      gpu_network_cache_->entity_mapping().size() !=
          network_entities_with_indices.size() ||
      !std::equal(gpu_network_cache_->entity_mapping().begin(),
                  gpu_network_cache_->entity_mapping().end(),
                  network_entities_with_indices.begin(),
                  network_entities_with_indices.end(),
                  [](uint32_t entity,
                     const std::pair<uint32_t, int> &entity_with_index) {
                    return entity == entity_with_index.first;
                  })) {
    MOONAI_PROFILE_SCOPE("gpu_cache_build");
    spdlog::debug("Rebuilding GPU network cache for {} network entities",
                  network_entities_with_indices.size());
    gpu_network_cache_->build_from(state.evolution.network_cache,
                                   network_entities_with_indices);
  }

  if (!gpu_network_cache_->launch_inference_async(
          gpu_batch.buffer().device_agent_sensor_inputs(),
          gpu_batch.buffer().device_agent_brain_outputs(),
          network_entities_with_indices.size(), gpu_batch.stream())) {
    gpu_batch.mark_error();
    return false;
  }

  return true;
}

} // namespace moonai
