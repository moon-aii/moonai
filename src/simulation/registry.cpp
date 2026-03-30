#include "simulation/registry.hpp"

#include <algorithm>

namespace moonai {

uint32_t Registry::create() {
  const uint32_t entity = static_cast<uint32_t>(size());
  resize(size() + 1);
  return entity;
}

void Registry::destroy(uint32_t entity) {
  if (valid(entity)) {
    vitals.alive[entity] = 0;
  }
}

bool Registry::valid(uint32_t entity) const {
  return entity != INVALID_ENTITY && entity < size();
}

void Registry::clear() {
  resize(0);
  next_agent_id_ = 1;
}

Registry::CompactionResult Registry::compact_dead() {
  CompactionResult result;

  std::size_t i = 0;
  while (i < size()) {
    if (vitals.alive[i] != 0) {
      ++i;
      continue;
    }

    const uint32_t removed = static_cast<uint32_t>(i);
    const std::size_t last = size() - 1;
    if (i != last) {
      const uint32_t moved_from = static_cast<uint32_t>(last);
      const uint32_t moved_to = static_cast<uint32_t>(i);
      swap_entities(i, last);
      result.moved.push_back({moved_from, moved_to});
    }

    result.removed.push_back(static_cast<uint32_t>(last));
    pop_back();
  }

  return result;
}

uint32_t Registry::find_by_agent_id(uint32_t agent_id) const {
  const auto it =
      std::find(identity.entity_id.begin(), identity.entity_id.end(), agent_id);
  if (it == identity.entity_id.end()) {
    return INVALID_ENTITY;
  }
  return static_cast<uint32_t>(std::distance(identity.entity_id.begin(), it));
}

void Registry::resize(std::size_t new_size) {
  positions.resize(new_size);
  motion.resize(new_size);
  vitals.resize(new_size);
  identity.resize(new_size);
  sensors.resize(new_size);
  stats.resize(new_size);
  brain.resize(new_size);
}

void Registry::swap_entities(std::size_t a, std::size_t b) {
  using std::swap;

  swap(positions.x[a], positions.x[b]);
  swap(positions.y[a], positions.y[b]);
  swap(motion.vel_x[a], motion.vel_x[b]);
  swap(motion.vel_y[a], motion.vel_y[b]);
  swap(motion.speed[a], motion.speed[b]);
  swap(vitals.energy[a], vitals.energy[b]);
  swap(vitals.age[a], vitals.age[b]);
  swap(vitals.alive[a], vitals.alive[b]);
  swap(identity.type[a], identity.type[b]);
  swap(identity.species_id[a], identity.species_id[b]);
  swap(identity.entity_id[a], identity.entity_id[b]);
  for (int i = 0; i < SensorSoA::INPUT_COUNT; ++i) {
    swap(sensors.inputs[a * SensorSoA::INPUT_COUNT + i],
         sensors.inputs[b * SensorSoA::INPUT_COUNT + i]);
  }
  swap(stats.kills[a], stats.kills[b]);
  swap(stats.food_eaten[a], stats.food_eaten[b]);
  swap(stats.distance_traveled[a], stats.distance_traveled[b]);
  swap(stats.offspring_count[a], stats.offspring_count[b]);
  swap(brain.decision_x[a], brain.decision_x[b]);
  swap(brain.decision_y[a], brain.decision_y[b]);
}

void Registry::pop_back() {
  const std::size_t new_size = size() - 1;
  positions.x.pop_back();
  positions.y.pop_back();
  motion.vel_x.pop_back();
  motion.vel_y.pop_back();
  motion.speed.pop_back();
  vitals.energy.pop_back();
  vitals.age.pop_back();
  vitals.alive.pop_back();
  identity.type.pop_back();
  identity.species_id.pop_back();
  identity.entity_id.pop_back();
  sensors.inputs.resize(new_size * SensorSoA::INPUT_COUNT);
  stats.kills.pop_back();
  stats.food_eaten.pop_back();
  stats.distance_traveled.pop_back();
  stats.offspring_count.pop_back();
  brain.decision_x.pop_back();
  brain.decision_y.pop_back();
}

} // namespace moonai
