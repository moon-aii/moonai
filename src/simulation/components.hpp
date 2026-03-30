#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moonai {

struct PositionSoA {
  std::vector<float> x;
  std::vector<float> y;

  void resize(std::size_t n) {
    x.resize(n);
    y.resize(n);
  }

  std::size_t size() const {
    return x.size();
  }
};

struct AgentSoA {
  static constexpr int INPUT_COUNT = 12;
  static constexpr int OUTPUT_COUNT = 2;

  std::vector<float> vel_x;
  std::vector<float> vel_y;
  std::vector<float> speed;
  std::vector<float> energy;
  std::vector<int> age;
  std::vector<uint8_t> alive;
  std::vector<uint32_t> species_id;
  std::vector<uint32_t> entity_id;
  std::vector<float> sensors;
  std::vector<float> decision_x;
  std::vector<float> decision_y;
  std::vector<float> distance_traveled;
  std::vector<int> offspring_count;

  void resize(std::size_t n) {
    vel_x.resize(n);
    vel_y.resize(n);
    speed.resize(n);
    energy.resize(n);
    age.resize(n);
    alive.resize(n);
    species_id.resize(n);
    entity_id.resize(n);
    sensors.resize(n * INPUT_COUNT);
    decision_x.resize(n);
    decision_y.resize(n);
    distance_traveled.resize(n);
    offspring_count.resize(n);
  }

  std::size_t size() const {
    return vel_x.size();
  }

  float *input_ptr(std::size_t entity) {
    return &sensors[entity * INPUT_COUNT];
  }

  const float *input_ptr(std::size_t entity) const {
    return &sensors[entity * INPUT_COUNT];
  }
};

struct PredatorSoA {
  std::vector<int> kills;

  void resize(std::size_t n) {
    kills.resize(n);
  }

  std::size_t size() const {
    return kills.size();
  }
};

struct PreySoA {
  std::vector<int> food_eaten;

  void resize(std::size_t n) {
    food_eaten.resize(n);
  }

  std::size_t size() const {
    return food_eaten.size();
  }
};

} // namespace moonai
