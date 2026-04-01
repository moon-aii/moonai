#include "simulation/simulation_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/evolution_manager.hpp"
#include "gpu/gpu_batch.hpp"
#include "simulation/simulation_step_systems.hpp"

#include <algorithm>
#include <cstring>
#include <spdlog/spdlog.h>

namespace moonai {

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
} // namespace

void SimulationManager::collect_gpu_step_events(AppState &state, const std::vector<uint8_t> &was_predator_alive,
                                                const std::vector<uint8_t> &was_prey_alive,
                                                const std::vector<uint8_t> &was_food_active) {
  auto &predator_buffer = gpu_batch_->predator_buffer();
  auto &prey_buffer = gpu_batch_->prey_buffer();
  auto &food_buffer = gpu_batch_->food_buffer();

  for (std::size_t food_idx = 0; food_idx < state.food.size(); ++food_idx) {
    const int prey_idx = food_buffer.host_consumed_by()[food_idx];
    if (!was_food_active[food_idx] || state.food.active[food_idx] || prey_idx < 0 ||
        static_cast<uint32_t>(prey_idx) >= state.prey.size()) {
      continue;
    }

    state.prey.consumption[prey_idx] += 1;
    ++state.runtime.step_events.food_eaten;
  }

  const uint32_t predator_count = static_cast<uint32_t>(state.predator.size());
  for (uint32_t predator_idx = 0; predator_idx < predator_count; ++predator_idx) {
    if (predator_buffer.host_kill_counts()[predator_idx] > 0) {
      state.predator.consumption[predator_idx] += static_cast<int>(predator_buffer.host_kill_counts()[predator_idx]);
    }
  }

  const uint32_t prey_count = static_cast<uint32_t>(state.prey.size());
  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    const int killer_idx = prey_buffer.host_claimed_by()[prey_idx];
    if (killer_idx >= 0 && static_cast<uint32_t>(killer_idx) < state.predator.size()) {
      ++state.runtime.step_events.kills;
    }
  }

  for (uint32_t predator_idx = 0; predator_idx < predator_count; ++predator_idx) {
    if (was_predator_alive[predator_idx] && state.predator.alive[predator_idx] == 0) {
      ++state.runtime.step_events.deaths;
    }
  }

  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    if (was_prey_alive[prey_idx] && state.prey.alive[prey_idx] == 0) {
      ++state.runtime.step_events.deaths;
    }
  }
}

void SimulationManager::ensure_gpu_capacity(std::size_t predator_count, std::size_t prey_count,
                                            std::size_t food_count) {
  if (!gpu_enabled_) {
    return;
  }

  const bool needs_batch = !gpu_batch_;
  const bool predators_exceeded = gpu_batch_ && predator_count > gpu_batch_->predator_capacity();
  const bool prey_exceeded = gpu_batch_ && prey_count > gpu_batch_->prey_capacity();
  const bool food_exceeded = gpu_batch_ && food_count > gpu_batch_->food_capacity();
  const bool needs_resize = predators_exceeded || prey_exceeded || food_exceeded;

  if (!needs_batch && !needs_resize) {
    return;
  }

  const std::size_t current_predator_capacity = gpu_batch_ ? gpu_batch_->predator_capacity() : 0;
  const std::size_t current_prey_capacity = gpu_batch_ ? gpu_batch_->prey_capacity() : 0;
  const std::size_t current_food_capacity = gpu_batch_ ? gpu_batch_->food_capacity() : 0;

  // Only resize buffers that exceeded their capacity
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

  gpu_batch_ = std::make_unique<gpu::GpuBatch>(new_predator_capacity, new_prey_capacity, new_food_capacity);
}

void SimulationManager::enable_gpu(bool enable) {
  gpu_enabled_ = enable;
  if (enable) {
    ensure_gpu_capacity(static_cast<std::size_t>(config_.predator_count), static_cast<std::size_t>(config_.prey_count),
                        static_cast<std::size_t>(config_.food_count));
  } else {
    gpu_batch_.reset();
    spdlog::info("GPU batch processing disabled");
  }
}

void SimulationManager::step_gpu(AppState &state, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_gpu");

  state.runtime.pending_predator_offspring.clear();
  state.runtime.pending_prey_offspring.clear();
  state.runtime.step_events.clear();

  if (!gpu_batch_ || !gpu_batch_->ok()) {
    spdlog::error("GPU batch not initialized or in error state, falling back to CPU");
    return step(state, evolution);
  }

  const std::size_t predator_count = state.predator.size();
  const std::size_t prey_count = state.prey.size();
  const std::size_t food_count = state.food.size();
  if (predator_count == 0 && prey_count == 0) {
    return;
  }

  std::vector<uint8_t> was_predator_alive;
  std::vector<uint8_t> was_prey_alive;
  std::vector<uint8_t> was_food_active;

  {
    MOONAI_PROFILE_SCOPE("gpu_ensure_capacity");
    ensure_gpu_capacity(predator_count, prey_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_pack_state");
    was_predator_alive = state.predator.alive;
    was_prey_alive = state.prey.alive;
    was_food_active = state.food.active;

    auto &predator_buffer = gpu_batch_->predator_buffer();
    auto &prey_buffer = gpu_batch_->prey_buffer();
    auto &food_buffer = gpu_batch_->food_buffer();

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

  gpu::GpuStepParams params;
  params.world_width = static_cast<float>(config_.grid_size);
  params.world_height = static_cast<float>(config_.grid_size);
  params.energy_drain_per_step = config_.energy_drain_per_step;
  params.vision_range = config_.vision_range;
  params.max_energy = static_cast<float>(config_.initial_energy);
  params.max_age = config_.max_steps;
  params.interaction_range = config_.interaction_range;
  params.energy_gain_from_food = static_cast<float>(config_.energy_gain_from_food);
  params.energy_gain_from_kill = static_cast<float>(config_.energy_gain_from_kill);
  params.predator_speed = config_.predator_speed;
  params.prey_speed = config_.prey_speed;

  {
    MOONAI_PROFILE_SCOPE("gpu_upload_enqueue");
    gpu_batch_->upload_async(predator_count, prey_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_sensors");
    gpu_batch_->launch_build_sensors_async(params, predator_count, prey_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_neural");
    if (!evolution.launch_gpu_neural(state, *gpu_batch_)) {
      MOONAI_PROFILE_SCOPE("cpu_fallback");
      spdlog::error("GPU neural inference failed, disabling GPU path and "
                    "retrying on CPU");
      gpu_enabled_ = false;
      gpu_batch_.reset();
      return step(state, evolution);
    }
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_step");
    gpu_batch_->launch_post_inference_async(params, predator_count, prey_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_download_enqueue");
    gpu_batch_->download_async(predator_count, prey_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_synchronize");
    gpu_batch_->synchronize();
  }

  if (!gpu_batch_->ok()) {
    MOONAI_PROFILE_SCOPE("cpu_fallback");
    spdlog::error("GPU step failed, disabling GPU path and retrying on CPU");
    gpu_enabled_ = false;
    gpu_batch_.reset();
    return step(state, evolution);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_apply_results");
    auto &predator_buffer = gpu_batch_->predator_buffer();
    auto &prey_buffer = gpu_batch_->prey_buffer();
    auto &food_buffer = gpu_batch_->food_buffer();

    if (predator_count > 0) {
      std::memcpy(state.predator.pos_x.data(), predator_buffer.host_positions_x(), predator_count * sizeof(float));
      std::memcpy(state.predator.pos_y.data(), predator_buffer.host_positions_y(), predator_count * sizeof(float));
      std::memcpy(state.predator.vel_x.data(), predator_buffer.host_velocities_x(), predator_count * sizeof(float));
      std::memcpy(state.predator.vel_y.data(), predator_buffer.host_velocities_y(), predator_count * sizeof(float));
      std::memcpy(state.predator.energy.data(), predator_buffer.host_energy(), predator_count * sizeof(float));
      std::memcpy(state.predator.age.data(), predator_buffer.host_age(), predator_count * sizeof(int));
      // Sensor inputs and brain outputs are device-only, not copied back to CPU
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
      // Sensor inputs and brain outputs are device-only, not copied back to CPU
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

  collect_gpu_step_events(state, was_predator_alive, was_prey_alive, was_food_active);
  compact_predators(state, evolution);
  compact_prey(state, evolution);
  refresh_world_state_after_step(state);
  state.runtime.pending_predator_offspring = find_predator_reproduction_pairs(state);
  state.runtime.pending_prey_offspring = find_prey_reproduction_pairs(state);
}

} // namespace moonai
