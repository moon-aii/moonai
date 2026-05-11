#include "simulation/simulation.hpp"

#include "core/profiler_macros.hpp"
#include "simulation/batch.hpp"

#include <algorithm>
#include <cstring>

#include <spdlog/spdlog.h>

namespace moonai::simulation {

namespace {

void collect_step_events(AppState &state, Batch &batch, const std::vector<uint8_t> &was_food_active) {
  MOONAI_PROFILE_SCOPE("collect_step_events");

  auto &prey_buffer = batch.prey_buffer();
  auto &food_buffer = batch.food_buffer();

  for (std::size_t food_idx = 0; food_idx < state.food.size(); ++food_idx) {
    const int prey_idx = food_buffer.host_consumed_by()[food_idx];
    if (was_food_active[food_idx] && !state.food.active[food_idx] && prey_idx >= 0 &&
        static_cast<uint32_t>(prey_idx) < state.prey.size()) {
      ++state.metrics.food_eaten;
    }
  }

  const uint32_t predator_count = static_cast<uint32_t>(state.predator.size());
  for (uint32_t predator_idx = 0; predator_idx < predator_count; ++predator_idx) {
    if (state.predator.alive[predator_idx] == 0) {
      ++state.metrics.predator_deaths;
    }
  }

  const uint32_t prey_count = static_cast<uint32_t>(state.prey.size());
  for (uint32_t prey_idx = 0; prey_idx < prey_count; ++prey_idx) {
    const int killer_idx = prey_buffer.host_claimed_by()[prey_idx];
    if (killer_idx >= 0 && static_cast<uint32_t>(killer_idx) < state.predator.size()) {
      ++state.metrics.kills;
    }
    if (state.prey.alive[prey_idx] == 0) {
      ++state.metrics.prey_deaths;
    }
  }
}

void pack_state(AppState &state, Batch &batch) {
  MOONAI_PROFILE_SCOPE("pack_state");

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

void apply_results(AppState &state, Batch &batch) {
  MOONAI_PROFILE_SCOPE("apply_results");

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

StepParams build_step_params(const SimulationConfig &config) {
  StepParams params;
  params.world_width = static_cast<float>(config.grid_size);
  params.world_height = static_cast<float>(config.grid_size);
  params.energy_drain_per_step = config.energy_drain_per_step;
  params.vision_range = config.vision_range;
  params.max_energy = config.max_energy;
  params.max_age = config.max_age;
  params.interaction_range = config.interaction_range;
  params.energy_gain_from_food = static_cast<float>(config.energy_gain_from_food);
  params.energy_gain_from_kill = static_cast<float>(config.energy_gain_from_kill);
  params.predator_speed = config.predator_speed;
  params.prey_speed = config.prey_speed;
  return params;
}

} // namespace

void initialize(AppState &state, const SimulationConfig &config) {
  state.food.initialize(config, state.runtime.rng);
}

bool prepare_step(AppState &state, const SimulationConfig &config) {
  MOONAI_PROFILE_SCOPE("prepare_step");

  state.step_buffers.was_food_active = state.food.active;

  const std::size_t predator_count = state.predator.size();
  const std::size_t prey_count = state.prey.size();
  const std::size_t food_count = state.food.size();

  if (!state.batch.ok()) {
    spdlog::error("Simulation batch is in an error state");
    return false;
  }

  state.batch.ensure_capacity(predator_count, prey_count, food_count);
  if (!state.batch.ok()) {
    spdlog::error("Failed to initialize the simulation batch");
    return false;
  }

  pack_state(state, state.batch);

  const StepParams params = build_step_params(config);

  state.batch.upload_async(predator_count, prey_count, food_count);
  state.batch.launch_build_sensors_async(params, predator_count, prey_count, food_count);

  return true;
}

bool resolve_step(AppState &state, const SimulationConfig &config) {
  MOONAI_PROFILE_SCOPE("resolve_step");

  if (!state.batch.ok()) {
    spdlog::error("Simulation batch is in an error state before resolve step");
    return false;
  }

  const std::size_t predator_count = state.predator.size();
  const std::size_t prey_count = state.prey.size();
  const std::size_t food_count = state.food.size();
  const StepParams params = build_step_params(config);

  state.batch.launch_post_inference_async(params, predator_count, prey_count, food_count);
  state.batch.download_async(predator_count, prey_count, food_count);
  state.batch.synchronize();

  if (!state.batch.ok()) {
    spdlog::error("Simulation step failed");
    return false;
  }

  apply_results(state, state.batch);
  collect_step_events(state, state.batch, state.step_buffers.was_food_active);

  return true;
}

void post_step(AppState &state, const SimulationConfig &config) {
  state.predator.compact();
  state.prey.compact();
  state.food.respawn_step(config, state.runtime.step, state.runtime.rng.seed());
}

} // namespace moonai::simulation
