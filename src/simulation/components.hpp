#pragma once
#include <cstdint>
#include <vector>

namespace moonai {

struct PositionSoA {
  std::vector<float> x;
  std::vector<float> y;

  void resize(size_t n) {
    x.resize(n);
    y.resize(n);
  }
  size_t size() const {
    return x.size();
  }
};

struct MotionSoA {
  std::vector<float> vel_x;
  std::vector<float> vel_y;
  std::vector<float> speed;

  void resize(size_t n) {
    vel_x.resize(n);
    vel_y.resize(n);
    speed.resize(n);
  }
  size_t size() const {
    return vel_x.size();
  }
};

struct VitalsSoA {
  std::vector<float> energy;
  std::vector<int> age;
  std::vector<uint8_t> alive;
  std::vector<int> reproduction_cooldown;

  void resize(size_t n) {
    energy.resize(n);
    age.resize(n);
    alive.resize(n);
    reproduction_cooldown.resize(n);
  }
  size_t size() const {
    return energy.size();
  }
};

struct IdentitySoA {
  static constexpr uint8_t TYPE_PREDATOR = 0;
  static constexpr uint8_t TYPE_PREY = 1;
  static constexpr uint8_t TYPE_FOOD = 2;

  std::vector<uint8_t> type;
  std::vector<uint32_t> species_id;
  std::vector<uint32_t> entity_id;

  void resize(size_t n) {
    type.resize(n);
    species_id.resize(n);
    entity_id.resize(n);
  }
  size_t size() const {
    return type.size();
  }
};

struct SensorSoA {
  static constexpr int INPUT_COUNT = 15;
  static constexpr int OUTPUT_COUNT = 2;

  std::vector<float> inputs;
  std::vector<float> outputs;

  void resize(size_t n) {
    inputs.resize(n * INPUT_COUNT);
    outputs.resize(n * OUTPUT_COUNT);
  }

  float *input_ptr(size_t entity) {
    return &inputs[entity * INPUT_COUNT];
  }
  float *output_ptr(size_t entity) {
    return &outputs[entity * OUTPUT_COUNT];
  }
  const float *input_ptr(size_t entity) const {
    return &inputs[entity * INPUT_COUNT];
  }
  const float *output_ptr(size_t entity) const {
    return &outputs[entity * OUTPUT_COUNT];
  }
};

struct StatsSoA {
  std::vector<int> kills;
  std::vector<int> food_eaten;
  std::vector<float> distance_traveled;
  std::vector<int> offspring_count;

  void resize(size_t n) {
    kills.resize(n);
    food_eaten.resize(n);
    distance_traveled.resize(n);
    offspring_count.resize(n);
  }
  size_t size() const {
    return kills.size();
  }
};

struct VisualSoA {
  std::vector<float> radius;
  std::vector<uint32_t> color_rgba;
  std::vector<uint8_t> shape_type;

  void resize(size_t n) {
    radius.resize(n);
    color_rgba.resize(n);
    shape_type.resize(n);
  }
};

struct BrainSoA {
  std::vector<float> decision_x;
  std::vector<float> decision_y;

  void resize(size_t n) {
    decision_x.resize(n);
    decision_y.resize(n);
  }
};

struct FoodStateSoA {
  std::vector<uint32_t> slot_index;
  std::vector<uint8_t> active;

  void resize(size_t n) {
    slot_index.resize(n);
    active.resize(n);
  }
  size_t size() const {
    return slot_index.size();
  }
};

} // namespace moonai