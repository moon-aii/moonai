#pragma once
#include "gpu/gpu_data_buffer.hpp"
#include "gpu/gpu_entity_mapping.hpp"
#include "simulation/entity.hpp"
#include "simulation/registry.hpp"

namespace moonai {
namespace gpu {

void pack_ecs_to_gpu(const Registry &registry, const GpuEntityMapping &mapping,
                     GpuDataBuffer &buffer);

void unpack_gpu_to_ecs(const GpuDataBuffer &buffer,
                       const GpuEntityMapping &mapping, Registry &registry);

uint32_t prepare_ecs_for_gpu(const Registry &registry,
                             GpuEntityMapping &mapping, GpuDataBuffer &buffer);

void apply_gpu_results(const GpuDataBuffer &buffer,
                       const GpuEntityMapping &mapping, Registry &registry);

} // namespace gpu
} // namespace moonai
