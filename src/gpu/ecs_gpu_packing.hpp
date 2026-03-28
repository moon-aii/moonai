#pragma once
#include "gpu/gpu_data_buffer.hpp"
#include "gpu/gpu_entity_mapping.hpp"
#include "simulation/step_state.hpp"

namespace moonai {
namespace gpu {

void pack_step_state_to_gpu(const PackedStepState &state,
                            const GpuEntityMapping &agent_mapping,
                            const GpuEntityMapping &food_mapping,
                            GpuDataBuffer &buffer);

void unpack_gpu_to_step_state(const GpuDataBuffer &buffer,
                              const GpuEntityMapping &agent_mapping,
                              const GpuEntityMapping &food_mapping,
                              PackedStepState &state);

void prepare_step_state_for_gpu(const PackedStepState &state,
                                GpuEntityMapping &agent_mapping,
                                GpuEntityMapping &food_mapping,
                                GpuDataBuffer &buffer);

void apply_gpu_results(const GpuDataBuffer &buffer,
                       const GpuEntityMapping &agent_mapping,
                       const GpuEntityMapping &food_mapping,
                       PackedStepState &state);

} // namespace gpu
} // namespace moonai
