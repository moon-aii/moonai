#include "gpu/ecs_gpu_packing.hpp"

#include <stdexcept>

namespace moonai {
namespace gpu {

void pack_step_state_to_gpu(const PackedStepState &state,
                            const GpuEntityMapping &agent_mapping,
                            const GpuEntityMapping &food_mapping,
                            GpuDataBuffer &buffer) {
  const uint32_t agent_count = agent_mapping.count();
  for (uint32_t gpu_idx = 0; gpu_idx < agent_count; ++gpu_idx) {
    buffer.host_agent_positions_x()[gpu_idx] = state.agents.pos_x[gpu_idx];
    buffer.host_agent_positions_y()[gpu_idx] = state.agents.pos_y[gpu_idx];
    buffer.host_agent_velocities_x()[gpu_idx] = state.agents.vel_x[gpu_idx];
    buffer.host_agent_velocities_y()[gpu_idx] = state.agents.vel_y[gpu_idx];
    buffer.host_agent_speed()[gpu_idx] = state.agents.speed[gpu_idx];
    buffer.host_agent_energy()[gpu_idx] = state.agents.energy[gpu_idx];
    buffer.host_agent_age()[gpu_idx] = state.agents.age[gpu_idx];
    buffer.host_agent_alive()[gpu_idx] = state.agents.alive[gpu_idx];
    buffer.host_agent_types()[gpu_idx] = state.agents.type[gpu_idx];
    buffer.host_agent_reproduction_cooldown()[gpu_idx] =
        state.agents.reproduction_cooldown[gpu_idx];
    buffer.host_agent_distance_traveled()[gpu_idx] =
        state.agents.distance_traveled[gpu_idx];
    buffer.host_agent_kill_counts()[gpu_idx] = 0;
    buffer.host_agent_killed_by()[gpu_idx] = -1;
  }

  const uint32_t food_count = food_mapping.count();
  for (uint32_t gpu_idx = 0; gpu_idx < food_count; ++gpu_idx) {
    buffer.host_food_positions_x()[gpu_idx] = state.foods.pos_x[gpu_idx];
    buffer.host_food_positions_y()[gpu_idx] = state.foods.pos_y[gpu_idx];
    buffer.host_food_active()[gpu_idx] = state.foods.active[gpu_idx];
    buffer.host_food_consumed_by()[gpu_idx] = -1;
  }
}

void unpack_gpu_to_step_state(const GpuDataBuffer &buffer,
                              const GpuEntityMapping &agent_mapping,
                              const GpuEntityMapping &food_mapping,
                              PackedStepState &state) {
  const uint32_t agent_count = agent_mapping.count();
  for (uint32_t gpu_idx = 0; gpu_idx < agent_count; ++gpu_idx) {
    state.agents.pos_x[gpu_idx] = buffer.host_agent_positions_x()[gpu_idx];
    state.agents.pos_y[gpu_idx] = buffer.host_agent_positions_y()[gpu_idx];
    state.agents.vel_x[gpu_idx] = buffer.host_agent_velocities_x()[gpu_idx];
    state.agents.vel_y[gpu_idx] = buffer.host_agent_velocities_y()[gpu_idx];
    state.agents.energy[gpu_idx] = buffer.host_agent_energy()[gpu_idx];
    state.agents.age[gpu_idx] = buffer.host_agent_age()[gpu_idx];
    state.agents.alive[gpu_idx] =
        static_cast<uint8_t>(buffer.host_agent_alive()[gpu_idx]);
    state.agents.reproduction_cooldown[gpu_idx] =
        buffer.host_agent_reproduction_cooldown()[gpu_idx];
    state.agents.distance_traveled[gpu_idx] =
        buffer.host_agent_distance_traveled()[gpu_idx];
    state.agents.kill_counts[gpu_idx] = buffer.host_agent_kill_counts()[gpu_idx];
    state.agents.killed_by[gpu_idx] = buffer.host_agent_killed_by()[gpu_idx];

    float *sensor_ptr = state.agents.sensor_ptr(gpu_idx);
    const float *host_sensor_ptr =
        buffer.host_agent_sensor_inputs() + gpu_idx * SensorSoA::INPUT_COUNT;
    for (int i = 0; i < SensorSoA::INPUT_COUNT; ++i) {
      sensor_ptr[i] = host_sensor_ptr[i];
    }

    float *brain_ptr = state.agents.brain_ptr(gpu_idx);
    const float *host_brain_ptr =
        buffer.host_agent_brain_outputs() + gpu_idx * SensorSoA::OUTPUT_COUNT;
    for (int i = 0; i < SensorSoA::OUTPUT_COUNT; ++i) {
      brain_ptr[i] = host_brain_ptr[i];
    }
  }

  const uint32_t food_count = food_mapping.count();
  for (uint32_t gpu_idx = 0; gpu_idx < food_count; ++gpu_idx) {
    state.foods.pos_x[gpu_idx] = buffer.host_food_positions_x()[gpu_idx];
    state.foods.pos_y[gpu_idx] = buffer.host_food_positions_y()[gpu_idx];
    state.foods.active[gpu_idx] =
        static_cast<uint8_t>(buffer.host_food_active()[gpu_idx]);
    state.foods.consumed_by[gpu_idx] = buffer.host_food_consumed_by()[gpu_idx];
  }
}

void prepare_step_state_for_gpu(const PackedStepState &state,
                                GpuEntityMapping &agent_mapping,
                                GpuEntityMapping &food_mapping,
                                GpuDataBuffer &buffer) {
  agent_mapping.build(state.agents.entities);
  food_mapping.build_count(state.foods.size());

  if (agent_mapping.count() > buffer.agent_capacity()) {
    throw std::runtime_error("GPU agent buffer capacity exceeded.");
  }
  if (food_mapping.count() > buffer.food_capacity()) {
    throw std::runtime_error("GPU food buffer capacity exceeded.");
  }

  pack_step_state_to_gpu(state, agent_mapping, food_mapping, buffer);
}

void apply_gpu_results(const GpuDataBuffer &buffer,
                       const GpuEntityMapping &agent_mapping,
                       const GpuEntityMapping &food_mapping,
                       PackedStepState &state) {
  unpack_gpu_to_step_state(buffer, agent_mapping, food_mapping, state);
}

} // namespace gpu
} // namespace moonai
