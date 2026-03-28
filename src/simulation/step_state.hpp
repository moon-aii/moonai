#pragma once

#include "simulation/components.hpp"
#include "simulation/entity.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace moonai {

struct PackedAgentStepState {
  std::vector<Entity> entities;
  std::vector<float> pos_x;
  std::vector<float> pos_y;
  std::vector<float> vel_x;
  std::vector<float> vel_y;
  std::vector<float> speed;
  std::vector<float> energy;
  std::vector<int> age;
  std::vector<uint8_t> alive;
  std::vector<uint8_t> was_alive;
  std::vector<uint8_t> type;
  std::vector<int> reproduction_cooldown;
  std::vector<float> distance_traveled;
  std::vector<uint32_t> kill_counts;
  std::vector<int> killed_by;
  std::vector<float> sensor_inputs;
  std::vector<float> brain_outputs;

  void resize(std::size_t count) {
    entities.resize(count);
    pos_x.resize(count);
    pos_y.resize(count);
    vel_x.resize(count);
    vel_y.resize(count);
    speed.resize(count);
    energy.resize(count);
    age.resize(count);
    alive.resize(count);
    was_alive.resize(count);
    type.resize(count);
    reproduction_cooldown.resize(count);
    distance_traveled.resize(count);
    kill_counts.resize(count);
    killed_by.resize(count);
    sensor_inputs.resize(count * SensorSoA::INPUT_COUNT);
    brain_outputs.resize(count * SensorSoA::OUTPUT_COUNT);
  }

  std::size_t size() const {
    return entities.size();
  }

  float *sensor_ptr(std::size_t idx) {
    return sensor_inputs.data() + idx * SensorSoA::INPUT_COUNT;
  }

  const float *sensor_ptr(std::size_t idx) const {
    return sensor_inputs.data() + idx * SensorSoA::INPUT_COUNT;
  }

  float *brain_ptr(std::size_t idx) {
    return brain_outputs.data() + idx * SensorSoA::OUTPUT_COUNT;
  }

  const float *brain_ptr(std::size_t idx) const {
    return brain_outputs.data() + idx * SensorSoA::OUTPUT_COUNT;
  }
};

struct PackedFoodStepState {
  std::vector<float> pos_x;
  std::vector<float> pos_y;
  std::vector<uint8_t> active;
  std::vector<uint8_t> was_active;
  std::vector<uint32_t> slot_index;
  std::vector<int> consumed_by;

  void resize(std::size_t count) {
    pos_x.resize(count);
    pos_y.resize(count);
    active.resize(count);
    was_active.resize(count);
    slot_index.resize(count);
    consumed_by.resize(count);
  }

  std::size_t size() const {
    return pos_x.size();
  }
};

struct PackedStepState {
  PackedAgentStepState agents;
  PackedFoodStepState foods;

  void clear_transients() {
    std::fill(agents.kill_counts.begin(), agents.kill_counts.end(), 0U);
    std::fill(agents.killed_by.begin(), agents.killed_by.end(), -1);
    std::fill(foods.consumed_by.begin(), foods.consumed_by.end(), -1);
  }
};

} // namespace moonai
