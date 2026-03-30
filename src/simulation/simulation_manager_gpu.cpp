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

void SimulationManager::collect_gpu_step_events(
    AppState &state, const std::vector<uint8_t> &was_predator_alive,
    const std::vector<uint8_t> &was_prey_alive,
    const std::vector<uint8_t> &was_food_active) {
  auto &predator_buffer = gpu_batch_->predator_buffer();
  auto &prey_buffer = gpu_batch_->prey_buffer();
  auto &food_buffer = gpu_batch_->food_buffer();

  for (std::size_t food_idx = 0; food_idx < state.food_store.size();
       ++food_idx) {
    const int prey_idx = food_buffer.host_consumed_by()[food_idx];
    if (!was_food_active[food_idx] || state.food_store.active[food_idx] ||
        prey_idx < 0 || static_cast<uint32_t>(prey_idx) >= state.prey.size()) {
      continue;
    }

    state.prey.prey.food_eaten[prey_idx] += 1;
    state.runtime.last_step_events.push_back(
        SimEvent{SimEvent::Food, state.prey.agents.entity_id[prey_idx], 0,
                 Vec2{state.prey.positions.x[prey_idx],
                      state.prey.positions.y[prey_idx]}});
  }

  const uint32_t predator_count = static_cast<uint32_t>(state.predators.size());
  for (uint32_t predator_idx = 0; predator_idx < predator_count;
       ++predator_idx) {
    if (predator_buffer.host_kill_counts()[predator_idx] > 0) {
      state.predators.predator.kills[predator_idx] +=
          static_cast<int>(predator_buffer.host_kill_counts()[predator_idx]);
    }
  }

  const uint32_t prey_count = static_cast<uint32_t>(state.prey.size());
  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    const int killer_idx = prey_buffer.host_claimed_by()[prey_idx];
    if (killer_idx >= 0 &&
        static_cast<uint32_t>(killer_idx) < state.predators.size()) {
      state.runtime.last_step_events.push_back(
          SimEvent{SimEvent::Kill, state.predators.agents.entity_id[killer_idx],
                   state.prey.agents.entity_id[prey_idx],
                   Vec2{state.prey.positions.x[prey_idx],
                        state.prey.positions.y[prey_idx]}});
    }
  }

  for (uint32_t predator_idx = 0; predator_idx < predator_count;
       ++predator_idx) {
    if (was_predator_alive[predator_idx] &&
        state.predators.agents.alive[predator_idx] == 0) {
      state.runtime.last_step_events.push_back(SimEvent{
          SimEvent::Death, state.predators.agents.entity_id[predator_idx],
          state.predators.agents.entity_id[predator_idx],
          Vec2{state.predators.positions.x[predator_idx],
               state.predators.positions.y[predator_idx]}});
    }
  }

  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    if (was_prey_alive[prey_idx] && state.prey.agents.alive[prey_idx] == 0) {
      state.runtime.last_step_events.push_back(
          SimEvent{SimEvent::Death, state.prey.agents.entity_id[prey_idx],
                   state.prey.agents.entity_id[prey_idx],
                   Vec2{state.prey.positions.x[prey_idx],
                        state.prey.positions.y[prey_idx]}});
    }
  }
}

void SimulationManager::ensure_gpu_capacity(std::size_t predator_count,
                                            std::size_t prey_count,
                                            std::size_t food_count) {
  if (!gpu_enabled_) {
    return;
  }

  const bool needs_batch = !gpu_batch_;
  const bool needs_resize =
      gpu_batch_ && (predator_count > gpu_batch_->predator_capacity() ||
                     prey_count > gpu_batch_->prey_capacity() ||
                     food_count > gpu_batch_->food_capacity());
  if (!needs_batch && !needs_resize) {
    return;
  }

  const std::size_t current_predator_capacity =
      gpu_batch_ ? gpu_batch_->predator_capacity() : 0;
  const std::size_t current_prey_capacity =
      gpu_batch_ ? gpu_batch_->prey_capacity() : 0;
  const std::size_t current_food_capacity =
      gpu_batch_ ? gpu_batch_->food_capacity() : 0;

  const std::size_t new_predator_capacity =
      std::max(predator_count, current_predator_capacity == 0
                                   ? predator_count
                                   : current_predator_capacity * 2);
  const std::size_t new_prey_capacity = std::max(
      prey_count,
      current_prey_capacity == 0 ? prey_count : current_prey_capacity * 2);
  const std::size_t new_food_capacity =
      std::max(food_count,
               current_food_capacity == 0 ? food_count : current_food_capacity);

  gpu_batch_ = std::make_unique<gpu::GpuBatch>(
      new_predator_capacity, new_prey_capacity, new_food_capacity);
  spdlog::info("GPU batch processing enabled with capacities {} predators / {} "
               "prey / {} food",
               new_predator_capacity, new_prey_capacity, new_food_capacity);
}

void SimulationManager::enable_gpu(bool enable) {
  gpu_enabled_ = enable;
  if (enable) {
    ensure_gpu_capacity(static_cast<std::size_t>(config_.predator_count),
                        static_cast<std::size_t>(config_.prey_count),
                        static_cast<std::size_t>(config_.food_count));
  } else {
    gpu_batch_.reset();
    spdlog::info("GPU batch processing disabled");
  }
}

void SimulationManager::step_gpu(AppState &state, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_gpu");

  state.runtime.last_step_events.clear();
  state.runtime.pending_predator_offspring.clear();
  state.runtime.pending_prey_offspring.clear();
  state.runtime.step_events.clear();

  if (!gpu_batch_ || !gpu_batch_->ok()) {
    spdlog::error(
        "GPU batch not initialized or in error state, falling back to CPU");
    return step(state, evolution);
  }

  const std::size_t predator_count = state.predators.size();
  const std::size_t prey_count = state.prey.size();
  const std::size_t food_count = state.food_store.size();
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
    was_predator_alive = state.predators.agents.alive;
    was_prey_alive = state.prey.agents.alive;
    was_food_active = state.food_store.active;

    auto &predator_buffer = gpu_batch_->predator_buffer();
    auto &prey_buffer = gpu_batch_->prey_buffer();
    auto &food_buffer = gpu_batch_->food_buffer();

    if (predator_count > 0) {
      std::memcpy(predator_buffer.host_positions_x(),
                  state.predators.positions.x.data(),
                  predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_positions_y(),
                  state.predators.positions.y.data(),
                  predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_velocities_x(),
                  state.predators.agents.vel_x.data(),
                  predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_velocities_y(),
                  state.predators.agents.vel_y.data(),
                  predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_speed(),
                  state.predators.agents.speed.data(),
                  predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_energy(),
                  state.predators.agents.energy.data(),
                  predator_count * sizeof(float));
      std::memcpy(predator_buffer.host_age(), state.predators.agents.age.data(),
                  predator_count * sizeof(int));
      std::memcpy(predator_buffer.host_distance_traveled(),
                  state.predators.agents.distance_traveled.data(),
                  predator_count * sizeof(float));
      for (uint32_t i = 0; i < static_cast<uint32_t>(predator_count); ++i) {
        predator_buffer.host_alive()[i] = state.predators.agents.alive[i];
      }
    }

    if (prey_count > 0) {
      std::memcpy(prey_buffer.host_positions_x(), state.prey.positions.x.data(),
                  prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_positions_y(), state.prey.positions.y.data(),
                  prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_velocities_x(),
                  state.prey.agents.vel_x.data(), prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_velocities_y(),
                  state.prey.agents.vel_y.data(), prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_speed(), state.prey.agents.speed.data(),
                  prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_energy(), state.prey.agents.energy.data(),
                  prey_count * sizeof(float));
      std::memcpy(prey_buffer.host_age(), state.prey.agents.age.data(),
                  prey_count * sizeof(int));
      std::memcpy(prey_buffer.host_distance_traveled(),
                  state.prey.agents.distance_traveled.data(),
                  prey_count * sizeof(float));
      for (uint32_t i = 0; i < static_cast<uint32_t>(prey_count); ++i) {
        prey_buffer.host_alive()[i] = state.prey.agents.alive[i];
      }
    }

    if (food_count > 0) {
      std::memcpy(food_buffer.host_positions_x(),
                  state.food_store.positions.x.data(),
                  food_count * sizeof(float));
      std::memcpy(food_buffer.host_positions_y(),
                  state.food_store.positions.y.data(),
                  food_count * sizeof(float));
      for (std::size_t i = 0; i < food_count; ++i) {
        food_buffer.host_active()[i] = state.food_store.active[i];
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
  params.energy_gain_from_food =
      static_cast<float>(config_.energy_gain_from_food);
  params.energy_gain_from_kill =
      static_cast<float>(config_.energy_gain_from_kill);

  {
    MOONAI_PROFILE_SCOPE("gpu_upload_enqueue");
    gpu_batch_->upload_async(predator_count, prey_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_sensors");
    gpu_batch_->launch_build_sensors_async(params, predator_count, prey_count,
                                           food_count);
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
    gpu_batch_->launch_post_inference_async(params, predator_count, prey_count,
                                            food_count);
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
      std::memcpy(state.predators.positions.x.data(),
                  predator_buffer.host_positions_x(),
                  predator_count * sizeof(float));
      std::memcpy(state.predators.positions.y.data(),
                  predator_buffer.host_positions_y(),
                  predator_count * sizeof(float));
      std::memcpy(state.predators.agents.vel_x.data(),
                  predator_buffer.host_velocities_x(),
                  predator_count * sizeof(float));
      std::memcpy(state.predators.agents.vel_y.data(),
                  predator_buffer.host_velocities_y(),
                  predator_count * sizeof(float));
      std::memcpy(state.predators.agents.energy.data(),
                  predator_buffer.host_energy(),
                  predator_count * sizeof(float));
      std::memcpy(state.predators.agents.age.data(), predator_buffer.host_age(),
                  predator_count * sizeof(int));
      std::memcpy(state.predators.agents.distance_traveled.data(),
                  predator_buffer.host_distance_traveled(),
                  predator_count * sizeof(float));
      std::memcpy(state.predators.agents.sensors.data(),
                  predator_buffer.host_sensor_inputs(),
                  predator_count * AgentSoA::INPUT_COUNT * sizeof(float));
      for (uint32_t i = 0; i < static_cast<uint32_t>(predator_count); ++i) {
        state.predators.agents.alive[i] =
            static_cast<uint8_t>(predator_buffer.host_alive()[i]);
        state.predators.agents.decision_x[i] =
            predator_buffer.host_brain_outputs()[i * AgentSoA::OUTPUT_COUNT];
        state.predators.agents.decision_y[i] =
            predator_buffer
                .host_brain_outputs()[i * AgentSoA::OUTPUT_COUNT + 1];
      }
    }

    if (prey_count > 0) {
      std::memcpy(state.prey.positions.x.data(), prey_buffer.host_positions_x(),
                  prey_count * sizeof(float));
      std::memcpy(state.prey.positions.y.data(), prey_buffer.host_positions_y(),
                  prey_count * sizeof(float));
      std::memcpy(state.prey.agents.vel_x.data(),
                  prey_buffer.host_velocities_x(), prey_count * sizeof(float));
      std::memcpy(state.prey.agents.vel_y.data(),
                  prey_buffer.host_velocities_y(), prey_count * sizeof(float));
      std::memcpy(state.prey.agents.energy.data(), prey_buffer.host_energy(),
                  prey_count * sizeof(float));
      std::memcpy(state.prey.agents.age.data(), prey_buffer.host_age(),
                  prey_count * sizeof(int));
      std::memcpy(state.prey.agents.distance_traveled.data(),
                  prey_buffer.host_distance_traveled(),
                  prey_count * sizeof(float));
      std::memcpy(state.prey.agents.sensors.data(),
                  prey_buffer.host_sensor_inputs(),
                  prey_count * AgentSoA::INPUT_COUNT * sizeof(float));
      for (uint32_t i = 0; i < static_cast<uint32_t>(prey_count); ++i) {
        state.prey.agents.alive[i] =
            static_cast<uint8_t>(prey_buffer.host_alive()[i]);
        state.prey.agents.decision_x[i] =
            prey_buffer.host_brain_outputs()[i * AgentSoA::OUTPUT_COUNT];
        state.prey.agents.decision_y[i] =
            prey_buffer.host_brain_outputs()[i * AgentSoA::OUTPUT_COUNT + 1];
      }
    }

    if (food_count > 0) {
      std::memcpy(state.food_store.positions.x.data(),
                  food_buffer.host_positions_x(), food_count * sizeof(float));
      std::memcpy(state.food_store.positions.y.data(),
                  food_buffer.host_positions_y(), food_count * sizeof(float));
      for (std::size_t i = 0; i < food_count; ++i) {
        state.food_store.active[i] =
            static_cast<uint8_t>(food_buffer.host_active()[i]);
      }
    }
  }

  collect_gpu_step_events(state, was_predator_alive, was_prey_alive,
                          was_food_active);
  compact_predators(state, evolution);
  compact_prey(state, evolution);
  refresh_world_state_after_step(state);
  state.runtime.pending_predator_offspring =
      find_predator_reproduction_pairs(state);
  state.runtime.pending_prey_offspring = find_prey_reproduction_pairs(state);
  simulation_detail::accumulate_events(state.runtime.step_events,
                                       state.runtime.last_step_events);
}

} // namespace moonai
