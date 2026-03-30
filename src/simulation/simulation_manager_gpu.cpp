#include "simulation/simulation_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/evolution_manager.hpp"
#include "gpu/gpu_batch.hpp"

#include <algorithm>
#include <cstring>
#include <spdlog/spdlog.h>

namespace moonai {

namespace {

void accumulate_events(EventCounters &counters,
                       const std::vector<SimEvent> &events) {
  for (const auto &event : events) {
    switch (event.type) {
      case SimEvent::Kill:
        ++counters.kills;
        break;
      case SimEvent::Food:
        ++counters.food_eaten;
        break;
      case SimEvent::Birth:
        ++counters.births;
        break;
      case SimEvent::Death:
        ++counters.deaths;
        break;
    }
  }
}

} // namespace

void SimulationManager::collect_gpu_step_events(
    AppState &state, const std::vector<uint8_t> &was_alive,
    const std::vector<uint8_t> &was_food_active) {
  auto &stats = state.registry.stats;
  const auto &positions = state.registry.positions;

  for (std::size_t food_idx = 0; food_idx < state.food_store.size();
       ++food_idx) {
    const int prey_idx = gpu_batch_->buffer().host_food_consumed_by()[food_idx];
    if (!was_food_active[food_idx] || state.food_store.active[food_idx] ||
        prey_idx < 0 ||
        static_cast<uint32_t>(prey_idx) >= state.registry.size()) {
      continue;
    }

    const uint32_t prey = static_cast<uint32_t>(prey_idx);
    stats.food_eaten[prey_idx] += 1;
    state.runtime.last_step_events.push_back(
        SimEvent{SimEvent::Food, prey, INVALID_ENTITY,
                 Vec2{positions.x[prey_idx], positions.y[prey_idx]}});
  }

  const uint32_t entity_count = static_cast<uint32_t>(state.registry.size());
  for (uint32_t agent_idx = 0; agent_idx < entity_count; ++agent_idx) {
    if (gpu_batch_->buffer().host_agent_kill_counts()[agent_idx] > 0) {
      stats.kills[agent_idx] += static_cast<int>(
          gpu_batch_->buffer().host_agent_kill_counts()[agent_idx]);
    }

    const int killer_idx =
        gpu_batch_->buffer().host_agent_killed_by()[agent_idx];
    if (killer_idx >= 0 &&
        static_cast<uint32_t>(killer_idx) < state.registry.size()) {
      state.runtime.last_step_events.push_back(
          SimEvent{SimEvent::Kill, static_cast<uint32_t>(killer_idx), agent_idx,
                   Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }

    if (was_alive[agent_idx] && state.registry.vitals.alive[agent_idx] == 0) {
      const uint32_t entity = agent_idx;
      state.runtime.last_step_events.push_back(
          SimEvent{SimEvent::Death, entity, entity,
                   Vec2{positions.x[agent_idx], positions.y[agent_idx]}});
    }
  }
}

void SimulationManager::ensure_gpu_capacity(std::size_t agent_count,
                                            std::size_t food_count) {
  if (!gpu_enabled_) {
    return;
  }

  const bool needs_batch = !gpu_batch_;
  const bool needs_resize =
      gpu_batch_ && (agent_count > gpu_batch_->agent_capacity() ||
                     food_count > gpu_batch_->food_capacity());
  if (!needs_batch && !needs_resize) {
    return;
  }

  const std::size_t current_agent_capacity =
      gpu_batch_ ? gpu_batch_->agent_capacity() : 0;
  const std::size_t current_food_capacity =
      gpu_batch_ ? gpu_batch_->food_capacity() : 0;
  const std::size_t new_agent_capacity = std::max(
      agent_count,
      current_agent_capacity == 0 ? agent_count : current_agent_capacity * 2);
  const std::size_t new_food_capacity =
      std::max(food_count,
               current_food_capacity == 0 ? food_count : current_food_capacity);

  gpu_batch_ =
      std::make_unique<gpu::GpuBatch>(new_agent_capacity, new_food_capacity);
  spdlog::info(
      "GPU batch processing enabled with capacities {} agents / {} food",
      new_agent_capacity, new_food_capacity);
}

void SimulationManager::enable_gpu(bool enable) {
  gpu_enabled_ = enable;
  if (enable) {
    ensure_gpu_capacity(
        static_cast<std::size_t>(config_.predator_count + config_.prey_count),
        static_cast<std::size_t>(config_.food_count));
  } else if (!enable) {
    gpu_batch_.reset();
    spdlog::info("GPU batch processing disabled");
  }
}

void SimulationManager::step_gpu(AppState &state, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_gpu");

  state.runtime.last_step_events.clear();
  state.runtime.pending_offspring.clear();
  state.runtime.step_events.clear();

  if (!gpu_batch_ || !gpu_batch_->ok()) {
    spdlog::error(
        "GPU batch not initialized or in error state, falling back to CPU");
    return step(state, evolution);
  }

  const std::size_t agent_count = state.registry.size();
  const std::size_t food_count = state.food_store.size();
  if (agent_count == 0) {
    return;
  }

  std::vector<uint8_t> was_alive;
  std::vector<uint8_t> was_food_active;
  {
    MOONAI_PROFILE_SCOPE("gpu_ensure_capacity");
    ensure_gpu_capacity(agent_count, food_count);
  }
  {
    MOONAI_PROFILE_SCOPE("gpu_pack_state");
    was_alive = state.registry.vitals.alive;
    was_food_active = state.food_store.active;

    auto &buffer = gpu_batch_->buffer();
    std::memcpy(buffer.host_agent_positions_x(),
                state.registry.positions.x.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_positions_y(),
                state.registry.positions.y.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_velocities_x(),
                state.registry.motion.vel_x.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_velocities_y(),
                state.registry.motion.vel_y.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_speed(), state.registry.motion.speed.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_energy(),
                state.registry.vitals.energy.data(),
                agent_count * sizeof(float));
    std::memcpy(buffer.host_agent_age(), state.registry.vitals.age.data(),
                agent_count * sizeof(int));
    const uint32_t packed_agent_count = static_cast<uint32_t>(agent_count);
    for (uint32_t i = 0; i < packed_agent_count; ++i) {
      buffer.host_agent_alive()[i] = state.registry.vitals.alive[i];
      buffer.host_agent_types()[i] = state.registry.identity.type[i];
    }
    std::memcpy(buffer.host_agent_distance_traveled(),
                state.registry.stats.distance_traveled.data(),
                agent_count * sizeof(float));

    std::memcpy(buffer.host_food_positions_x(), state.food_store.pos_x.data(),
                food_count * sizeof(float));
    std::memcpy(buffer.host_food_positions_y(), state.food_store.pos_y.data(),
                food_count * sizeof(float));
    for (std::size_t i = 0; i < food_count; ++i) {
      buffer.host_food_active()[i] = state.food_store.active[i];
      buffer.host_food_consumed_by()[i] = -1;
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
    gpu_batch_->upload_async(agent_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_sensors");
    gpu_batch_->launch_build_sensors_async(params, agent_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_launch_neural");
    if (!evolution.launch_gpu_neural(state, *gpu_batch_, agent_count)) {
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
    gpu_batch_->launch_post_inference_async(params, agent_count, food_count);
  }

  {
    MOONAI_PROFILE_SCOPE("gpu_download_enqueue");
    gpu_batch_->download_async(agent_count, food_count);
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
    auto &buffer = gpu_batch_->buffer();
    std::memcpy(state.registry.positions.x.data(),
                buffer.host_agent_positions_x(), agent_count * sizeof(float));
    std::memcpy(state.registry.positions.y.data(),
                buffer.host_agent_positions_y(), agent_count * sizeof(float));
    std::memcpy(state.registry.motion.vel_x.data(),
                buffer.host_agent_velocities_x(), agent_count * sizeof(float));
    std::memcpy(state.registry.motion.vel_y.data(),
                buffer.host_agent_velocities_y(), agent_count * sizeof(float));
    std::memcpy(state.registry.vitals.energy.data(),
                buffer.host_agent_energy(), agent_count * sizeof(float));
    std::memcpy(state.registry.vitals.age.data(), buffer.host_agent_age(),
                agent_count * sizeof(int));
    std::memcpy(state.registry.stats.distance_traveled.data(),
                buffer.host_agent_distance_traveled(),
                agent_count * sizeof(float));
    std::memcpy(state.registry.sensors.inputs.data(),
                buffer.host_agent_sensor_inputs(),
                agent_count * SensorSoA::INPUT_COUNT * sizeof(float));
    const uint32_t packed_agent_count = static_cast<uint32_t>(agent_count);
    for (uint32_t i = 0; i < packed_agent_count; ++i) {
      state.registry.vitals.alive[i] =
          static_cast<uint8_t>(buffer.host_agent_alive()[i]);
      state.registry.brain.decision_x[i] =
          buffer.host_agent_brain_outputs()[i * SensorSoA::OUTPUT_COUNT];
      state.registry.brain.decision_y[i] =
          buffer.host_agent_brain_outputs()[i * SensorSoA::OUTPUT_COUNT + 1];
    }
    for (std::size_t i = 0; i < food_count; ++i) {
      state.food_store.pos_x[i] = buffer.host_food_positions_x()[i];
      state.food_store.pos_y[i] = buffer.host_food_positions_y()[i];
      state.food_store.active[i] =
          static_cast<uint8_t>(buffer.host_food_active()[i]);
    }
  }

  collect_gpu_step_events(state, was_alive, was_food_active);
  compact_registry(state, evolution);
  refresh_world_state_after_step(state);
  state.runtime.pending_offspring = find_reproduction_pairs(state);
  accumulate_events(state.runtime.step_events, state.runtime.last_step_events);
}

} // namespace moonai
