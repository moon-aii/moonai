#include "core/types.hpp"
#include "gpu/ecs_gpu_packing.hpp"
#include <stdexcept>

namespace moonai {
namespace gpu {

void pack_ecs_to_gpu(const Registry &registry, const GpuEntityMapping &mapping,
                     GpuDataBuffer &buffer) {
  const uint32_t count = mapping.count();
  if (count == 0) {
    return;
  }

  // Get ECS component arrays
  const PositionSoA &positions = registry.positions();
  const MotionSoA &motion = registry.motion();
  const VitalsSoA &vitals = registry.vitals();
  const IdentitySoA &identity = registry.identity();

  // Pack living entities into contiguous GPU buffer
  for (uint32_t gpu_idx = 0; gpu_idx < count; ++gpu_idx) {
    Entity entity = mapping.entity_at(gpu_idx);
    size_t ecs_idx = registry.index_of(entity);

    // Copy component data to GPU buffer
    buffer.host_positions_x()[gpu_idx] = positions.x[ecs_idx];
    buffer.host_positions_y()[gpu_idx] = positions.y[ecs_idx];
    buffer.host_velocities_x()[gpu_idx] = motion.vel_x[ecs_idx];
    buffer.host_velocities_y()[gpu_idx] = motion.vel_y[ecs_idx];
    buffer.host_energy()[gpu_idx] = vitals.energy[ecs_idx];
    buffer.host_age()[gpu_idx] = static_cast<float>(vitals.age[ecs_idx]);
    buffer.host_alive()[gpu_idx] = vitals.alive[ecs_idx];
    buffer.host_types()[gpu_idx] = static_cast<uint8_t>(identity.type[ecs_idx]);
    buffer.host_species_ids()[gpu_idx] =
        static_cast<uint32_t>(identity.species_id[ecs_idx]);
  }
}

void unpack_gpu_to_ecs(const GpuDataBuffer &buffer,
                       const GpuEntityMapping &mapping, Registry &registry) {
  const uint32_t count = mapping.count();
  if (count == 0) {
    return;
  }

  // Get ECS component arrays for modification
  VitalsSoA &vitals = registry.vitals();
  MotionSoA &motion = registry.motion();
  PositionSoA &positions = registry.positions();

  // Copy results back to ECS using reverse mapping
  for (uint32_t gpu_idx = 0; gpu_idx < count; ++gpu_idx) {
    Entity entity = mapping.entity_at(gpu_idx);
    size_t ecs_idx = registry.index_of(entity);

    // Update ECS from GPU results
    vitals.energy[ecs_idx] = buffer.host_outputs_energy()[gpu_idx];
    vitals.alive[ecs_idx] = buffer.host_outputs_alive()[gpu_idx];
    motion.vel_x[ecs_idx] = buffer.host_outputs_velocities_x()[gpu_idx];
    motion.vel_y[ecs_idx] = buffer.host_outputs_velocities_y()[gpu_idx];

    // Note: Position updates are in-place on device, so we need to sync
    // positions back as well (done in separate position download)
    positions.x[ecs_idx] = buffer.host_positions_x()[gpu_idx];
    positions.y[ecs_idx] = buffer.host_positions_y()[gpu_idx];
  }
}

uint32_t prepare_ecs_for_gpu(const Registry &registry,
                             GpuEntityMapping &mapping, GpuDataBuffer &buffer) {
  // Build mapping from living entities
  const std::vector<Entity> &living = registry.living_entities();
  mapping.build(living);

  // Resize mapping if needed
  if (mapping.count() > buffer.capacity()) {
    // This shouldn't happen if buffer was created with max capacity
    // But we handle it gracefully
    throw std::runtime_error(
        "GPU buffer capacity exceeded. Increase max_entities.");
  }

  // Pack data into buffer
  pack_ecs_to_gpu(registry, mapping, buffer);

  return mapping.count();
}

void apply_gpu_results(const GpuDataBuffer &buffer,
                       const GpuEntityMapping &mapping, Registry &registry) {
  unpack_gpu_to_ecs(buffer, mapping, registry);
}

} // namespace gpu
} // namespace moonai
