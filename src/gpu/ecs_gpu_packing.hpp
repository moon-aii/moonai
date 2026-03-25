#pragma once
#include "gpu/gpu_data_buffer.hpp"
#include "gpu/gpu_entity_mapping.hpp"
#include "simulation/entity.hpp"
#include "simulation/registry.hpp"

namespace moonai {
namespace gpu {

/**
 * @brief Pack ECS component data into GPU buffers via entity mapping
 *
 * Performs scatter-gather: copies living entities from sparse ECS arrays
 * into contiguous GPU buffers. This is O(N) where N is the number of
 * living entities.
 *
 * @param registry ECS registry with agent data
 * @param mapping Entity → GPU index mapping (already built)
 * @param buffer GPU data buffer to fill
 */
void pack_ecs_to_gpu(const Registry &registry, const GpuEntityMapping &mapping,
                     GpuDataBuffer &buffer);

/**
 * @brief Unpack GPU buffer results back into ECS
 *
 * Copies kernel results from contiguous GPU buffers back to sparse ECS
 * arrays using the reverse mapping. This is O(N) where N is the number
 * of living entities.
 *
 * @param buffer GPU data buffer with results
 * @param mapping GPU index → Entity mapping
 * @param registry ECS registry to update
 */
void unpack_gpu_to_ecs(const GpuDataBuffer &buffer,
                       const GpuEntityMapping &mapping, Registry &registry);

/**
 * @brief Convenience: build mapping + pack in one call
 *
 * @param registry ECS registry
 * @param mapping Mapping object (will be resized and built)
 * @param buffer GPU buffer (will be filled)
 * @return Number of entities packed
 */
uint32_t prepare_ecs_for_gpu(const Registry &registry,
                             GpuEntityMapping &mapping, GpuDataBuffer &buffer);

/**
 * @brief Convenience: download + unpack in one call
 *
 * @param buffer GPU buffer with results
 * @param mapping Entity mapping
 * @param registry ECS registry to update
 */
void apply_gpu_results(const GpuDataBuffer &buffer,
                       const GpuEntityMapping &mapping, Registry &registry);

} // namespace gpu
} // namespace moonai
