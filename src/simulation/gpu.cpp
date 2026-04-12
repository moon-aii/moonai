#include "simulation/gpu.hpp"

#include "core/profiler_macros.hpp"

#include <algorithm>
#include <cstring>
#include <spdlog/spdlog.h>

namespace moonai::simulation::gpu {

namespace {

// Round up to the next power of 2 for GPU buffer allocation
inline std::size_t next_power_of_2(std::size_t n) {
  if (n == 0)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}

void collect_step_events(AppState &state, moonai::gpu::GpuBatch &batch, const std::vector<uint8_t> &was_food_active) {
  MOONAI_PROFILE_SCOPE("collect_gpu_step_events");

  auto &predator_buffer = batch.predator_buffer();
  auto &prey_buffer = batch.prey_buffer();
  auto &food_buffer = batch.food_buffer();

  for (std::size_t food_idx = 0; food_idx < state.food.size(); ++food_idx) {
    const int prey_idx = food_buffer.host_consumed_by()[food_idx];
    if (was_food_active[food_idx] && !state.food.active[food_idx] && prey_idx >= 0 &&
        static_cast<uint32_t>(prey_idx) < state.prey.size()) {
      ++state.prey.consumption[prey_idx];
      ++state.metrics.step_delta.food_eaten;
    }
  }

  const uint32_t predator_count = static_cast<uint32_t>(state.predator.size());
  for (uint32_t predator_idx = 0; predator_idx < predator_count; ++predator_idx) {
    if (predator_buffer.host_kill_counts()[predator_idx] > 0) {
      state.predator.consumption[predator_idx] += static_cast<int>(predator_buffer.host_kill_counts()[predator_idx]);
    }
    if (state.predator.alive[predator_idx] == 0) {
      ++state.metrics.step_delta.deaths;
    }
  }

  const uint32_t prey_count = static_cast<uint32_t>(state.prey.size());
  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    const int killer_idx = prey_buffer.host_claimed_by()[prey_idx];
    if (killer_idx >= 0 && static_cast<uint32_t>(killer_idx) < state.predator.size()) {
      ++state.metrics.step_delta.kills;
    }
    if (state.prey.alive[prey_idx] == 0) {
      ++state.metrics.step_delta.deaths;
    }
  }
}

void pack_state(AppState &state, moonai::gpu::GpuBatch &batch) {
  MOONAI_PROFILE_SCOPE("gpu_pack_state");

  auto &predator_buffer = batch.predator_buffer();
  auto &prey_buffer = batch.prey_buffer();
  auto &food_buffer = batch.food_buffer();

  const std::size_t predator_count = state.predator.size();
  const std::size_t prey_count = state.prey.size();
  const std::size_t food_count = state.food.size();

  if (predator_count > 0) {
    std::memcpy(predator_buffer.host_positions_x(), state.predator.pos_x.data(), predator_count * sizeof(float));
    std::memcpy(predator_buffer.host_positions_y(), state.predator.pos_y.data(), predator_count * sizeof(float));
    std::memcpy(predator_buffer.host_velocities_x(), state.predator.vel_x.data(), predator_count * sizeof(float));
    std::memcpy(predator_buffer.host_velocities_y(), state.predator.vel_y.data(), predator_count * sizeof(float));
    std::memcpy(predator_buffer.host_energy(), state.predator.energy.data(), predator_count * sizeof(float));
    std::memcpy(predator_buffer.host_age(), state.predator.age.data(), predator_count * sizeof(int));
    for (uint32_t i = 0; i < static_cast<uint32_t>(predator_count); ++i) {
      predator_buffer.host_alive()[i] = state.predator.alive[i];
    }
  }

  if (prey_count > 0) {
    std::memcpy(prey_buffer.host_positions_x(), state.prey.pos_x.data(), prey_count * sizeof(float));
    std::memcpy(prey_buffer.host_positions_y(), state.prey.pos_y.data(), prey_count * sizeof(float));
    std::memcpy(prey_buffer.host_velocities_x(), state.prey.vel_x.data(), prey_count * sizeof(float));
    std::memcpy(prey_buffer.host_velocities_y(), state.prey.vel_y.data(), prey_count * sizeof(float));
    std::memcpy(prey_buffer.host_energy(), state.prey.energy.data(), prey_count * sizeof(float));
    std::memcpy(prey_buffer.host_age(), state.prey.age.data(), prey_count * sizeof(int));
    for (uint32_t i = 0; i < static_cast<uint32_t>(prey_count); ++i) {
      prey_buffer.host_alive()[i] = state.prey.alive[i];
    }
  }

  if (food_count > 0) {
    std::memcpy(food_buffer.host_positions_x(), state.food.pos_x.data(), food_count * sizeof(float));
    std::memcpy(food_buffer.host_positions_y(), state.food.pos_y.data(), food_count * sizeof(float));
    for (std::size_t i = 0; i < food_count; ++i) {
      food_buffer.host_active()[i] = state.food.active[i];
      food_buffer.host_consumed_by()[i] = -1;
    }
  }
}

void apply_results(AppState &state, moonai::gpu::GpuBatch &batch) {
  MOONAI_PROFILE_SCOPE("gpu_apply_results");

  auto &predator_buffer = batch.predator_buffer();
  auto &prey_buffer = batch.prey_buffer();
  auto &food_buffer = batch.food_buffer();

  const std::size_t predator_count = state.predator.size();
  const std::size_t prey_count = state.prey.size();
  const std::size_t food_count = state.food.size();

  if (predator_count > 0) {
    std::memcpy(state.predator.pos_x.data(), predator_buffer.host_positions_x(), predator_count * sizeof(float));
    std::memcpy(state.predator.pos_y.data(), predator_buffer.host_positions_y(), predator_count * sizeof(float));
    std::memcpy(state.predator.vel_x.data(), predator_buffer.host_velocities_x(), predator_count * sizeof(float));
    std::memcpy(state.predator.vel_y.data(), predator_buffer.host_velocities_y(), predator_count * sizeof(float));
    std::memcpy(state.predator.energy.data(), predator_buffer.host_energy(), predator_count * sizeof(float));
    std::memcpy(state.predator.age.data(), predator_buffer.host_age(), predator_count * sizeof(int));
    for (uint32_t i = 0; i < static_cast<uint32_t>(predator_count); ++i) {
      state.predator.alive[i] = static_cast<uint8_t>(predator_buffer.host_alive()[i]);
    }
  }

  if (prey_count > 0) {
    std::memcpy(state.prey.pos_x.data(), prey_buffer.host_positions_x(), prey_count * sizeof(float));
    std::memcpy(state.prey.pos_y.data(), prey_buffer.host_positions_y(), prey_count * sizeof(float));
    std::memcpy(state.prey.vel_x.data(), prey_buffer.host_velocities_x(), prey_count * sizeof(float));
    std::memcpy(state.prey.vel_y.data(), prey_buffer.host_velocities_y(), prey_count * sizeof(float));
    std::memcpy(state.prey.energy.data(), prey_buffer.host_energy(), prey_count * sizeof(float));
    std::memcpy(state.prey.age.data(), prey_buffer.host_age(), prey_count * sizeof(int));
    for (uint32_t i = 0; i < static_cast<uint32_t>(prey_count); ++i) {
      state.prey.alive[i] = static_cast<uint8_t>(prey_buffer.host_alive()[i]);
    }
  }

  if (food_count > 0) {
    std::memcpy(state.food.pos_x.data(), food_buffer.host_positions_x(), food_count * sizeof(float));
    std::memcpy(state.food.pos_y.data(), food_buffer.host_positions_y(), food_count * sizeof(float));
    for (std::size_t i = 0; i < food_count; ++i) {
      state.food.active[i] = static_cast<uint8_t>(food_buffer.host_active()[i]);
    }
  }
}

void ensure_capacity(std::unique_ptr<moonai::gpu::GpuBatch> &batch, std::size_t predator_count, std::size_t prey_count,
                     std::size_t food_count) {
  MOONAI_PROFILE_SCOPE("gpu_ensure_capacity");

  const bool needs_batch = !batch;
  const bool predators_exceeded = batch && predator_count > batch->predator_capacity();
  const bool prey_exceeded = batch && prey_count > batch->prey_capacity();
  const bool food_exceeded = batch && food_count > batch->food_capacity();
  const bool needs_resize = predators_exceeded || prey_exceeded || food_exceeded;

  if (!needs_batch && !needs_resize) {
    return;
  }

  const std::size_t current_predator_capacity = batch ? batch->predator_capacity() : 0;
  const std::size_t current_prey_capacity = batch ? batch->prey_capacity() : 0;
  const std::size_t current_food_capacity = batch ? batch->food_capacity() : 0;

  const std::size_t new_predator_capacity =
      needs_batch || predators_exceeded
          ? next_power_of_2(current_predator_capacity == 0 ? predator_count
                                                           : std::max(predator_count, current_predator_capacity * 2))
          : current_predator_capacity;
  const std::size_t new_prey_capacity =
      needs_batch || prey_exceeded
          ? next_power_of_2(current_prey_capacity == 0 ? prey_count : std::max(prey_count, current_prey_capacity * 2))
          : current_prey_capacity;
  const std::size_t new_food_capacity =
      needs_batch || food_exceeded
          ? next_power_of_2(current_food_capacity == 0 ? food_count : std::max(food_count, current_food_capacity * 2))
          : current_food_capacity;

  batch = std::make_unique<moonai::gpu::GpuBatch>(new_predator_capacity, new_prey_capacity, new_food_capacity);
}

moonai::gpu::GpuStepParams build_step_params(const SimulationConfig &config) {
  moonai::gpu::GpuStepParams params;
  params.world_width = static_cast<float>(config.grid_size);
  params.world_height = static_cast<float>(config.grid_size);
  params.energy_drain_per_step = config.energy_drain_per_step;
  params.vision_range = config.vision_range;
  params.max_energy = static_cast<float>(config.initial_energy);
  params.max_age = config.max_steps;
  params.interaction_range = config.interaction_range;
  params.energy_gain_from_food = static_cast<float>(config.energy_gain_from_food);
  params.energy_gain_from_kill = static_cast<float>(config.energy_gain_from_kill);
  params.predator_speed = config.predator_speed;
  params.prey_speed = config.prey_speed;
  return params;
}

} // namespace

bool prepare_step(AppState &state, const SimulationConfig &config) {
  MOONAI_PROFILE_SCOPE("simulation_gpu");

  state.step_buffers.was_food_active = state.food.active;

  const std::size_t predator_count = state.predator.size();
  const std::size_t prey_count = state.prey.size();
  const std::size_t food_count = state.food.size();

  if (state.gpu_batch && !state.gpu_batch->ok()) {
    spdlog::error("GPU batch in error state, disabling GPU path");
    state.runtime.gpu_enabled = false;
    state.gpu_batch.reset();
    return false;
  }

  ensure_capacity(state.gpu_batch, predator_count, prey_count, food_count);
  pack_state(state, *state.gpu_batch);

  const moonai::gpu::GpuStepParams params = build_step_params(config);

  state.gpu_batch->upload_async(predator_count, prey_count, food_count);
  state.gpu_batch->launch_build_sensors_async(params, predator_count, prey_count, food_count);

  return true;
}

bool resolve_step(AppState &state, const SimulationConfig &config) {
  MOONAI_PROFILE_SCOPE("simulation_gpu_resolve");

  if (!state.gpu_batch) {
    spdlog::error("GPU batch is not initialized for resolve step");
    state.runtime.gpu_enabled = false;
    return false;
  }

  const std::size_t predator_count = state.predator.size();
  const std::size_t prey_count = state.prey.size();
  const std::size_t food_count = state.food.size();
  const moonai::gpu::GpuStepParams params = build_step_params(config);

  state.gpu_batch->launch_post_inference_async(params, predator_count, prey_count, food_count);
  state.gpu_batch->download_async(predator_count, prey_count, food_count);
  state.gpu_batch->synchronize();

  if (!state.gpu_batch->ok()) {
    spdlog::error("GPU step failed, disabling GPU path");
    state.runtime.gpu_enabled = false;
    state.gpu_batch.reset();
    return false;
  }

  apply_results(state, *state.gpu_batch);
  collect_step_events(state, *state.gpu_batch, state.step_buffers.was_food_active);

  return true;
}

} // namespace moonai::simulation::gpu
