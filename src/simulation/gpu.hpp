#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "gpu/gpu_batch.hpp"

#include <cstddef>
#include <memory>

namespace moonai {

class EvolutionManager;

namespace gpu {

void step(AppState &state, EvolutionManager &evolution, std::unique_ptr<gpu::GpuBatch> &batch,
          const SimulationConfig &config);

void ensure_capacity(std::unique_ptr<gpu::GpuBatch> &batch, std::size_t predator_count, std::size_t prey_count,
                     std::size_t food_count);

void disable(std::unique_ptr<gpu::GpuBatch> &batch);

} // namespace gpu

} // namespace moonai