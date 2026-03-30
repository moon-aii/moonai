#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "gpu/gpu_batch.hpp"
#include "gpu/gpu_network_cache.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace moonai {

namespace {

bool launch_population_gpu_neural(PopulationEvolutionState &population,
                                  gpu::GpuNetworkCache &gpu_cache,
                                  gpu::GpuPopulationBuffer &buffer,
                                  std::size_t count, cudaStream_t stream) {
  std::vector<std::pair<uint32_t, int>> network_entities_with_indices;
  network_entities_with_indices.reserve(count);

  const uint32_t entity_count = static_cast<uint32_t>(count);
  for (uint32_t entity = 0; entity < entity_count; ++entity) {
    if (population.network_cache.has(entity)) {
      network_entities_with_indices.emplace_back(entity,
                                                 static_cast<int>(entity));
    }
  }

  if (network_entities_with_indices.empty()) {
    return true;
  }

  if (gpu_cache.is_dirty() ||
      gpu_cache.entity_mapping().size() !=
          network_entities_with_indices.size() ||
      !std::equal(gpu_cache.entity_mapping().begin(),
                  gpu_cache.entity_mapping().end(),
                  network_entities_with_indices.begin(),
                  [](uint32_t entity,
                     const std::pair<uint32_t, int> &entity_with_index) {
                    return entity == entity_with_index.first;
                  })) {
    gpu_cache.build_from(population.network_cache,
                         network_entities_with_indices);
  }

  return gpu_cache.launch_inference_async(
      buffer.device_sensor_inputs(), buffer.device_brain_outputs(),
      network_entities_with_indices.size(), stream);
}

} // namespace

void EvolutionManager::enable_gpu(bool use_gpu) {
  use_gpu_ = use_gpu;
  if (use_gpu_) {
    if (!predator_gpu_network_cache_) {
      predator_gpu_network_cache_ = std::make_unique<gpu::GpuNetworkCache>();
      predator_gpu_network_cache_->invalidate();
    }
    if (!prey_gpu_network_cache_) {
      prey_gpu_network_cache_ = std::make_unique<gpu::GpuNetworkCache>();
      prey_gpu_network_cache_->invalidate();
    }
    spdlog::info("GPU neural inference enabled");
  } else {
    predator_gpu_network_cache_.reset();
    prey_gpu_network_cache_.reset();
    spdlog::info("GPU neural inference disabled");
  }
}

bool EvolutionManager::launch_gpu_neural(AppState &state,
                                         gpu::GpuBatch &gpu_batch) {
  MOONAI_PROFILE_SCOPE("gpu_neural", gpu_batch.stream());

  if (!predator_gpu_network_cache_ || !prey_gpu_network_cache_) {
    spdlog::error("GPU neural caches not initialized");
    return false;
  }

  if (!launch_population_gpu_neural(
          state.evolution.predators, *predator_gpu_network_cache_,
          gpu_batch.predator_buffer(), state.predators.size(),
          gpu_batch.stream())) {
    gpu_batch.mark_error();
    return false;
  }

  if (!launch_population_gpu_neural(
          state.evolution.prey, *prey_gpu_network_cache_,
          gpu_batch.prey_buffer(), state.prey.size(), gpu_batch.stream())) {
    gpu_batch.mark_error();
    return false;
  }

  return true;
}

} // namespace moonai
