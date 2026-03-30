#include "simulation/registry.hpp"

#include "core/deterministic_respawn.hpp"

#include <algorithm>

namespace moonai {

namespace {

template <typename SpecificSoA>
void resize_registry(PositionSoA &positions, AgentSoA &agents,
                     SpecificSoA &specific, std::size_t new_size) {
  positions.resize(new_size);
  agents.resize(new_size);
  specific.resize(new_size);
}

template <typename SpecificSoA>
std::size_t registry_size(const PositionSoA &positions, const AgentSoA &agents,
                          const SpecificSoA &specific) {
  (void)agents;
  (void)specific;
  return positions.size();
}

void swap_agent_fields(PositionSoA &positions, AgentSoA &agents, std::size_t a,
                       std::size_t b) {
  using std::swap;

  swap(positions.x[a], positions.x[b]);
  swap(positions.y[a], positions.y[b]);
  swap(agents.vel_x[a], agents.vel_x[b]);
  swap(agents.vel_y[a], agents.vel_y[b]);
  swap(agents.speed[a], agents.speed[b]);
  swap(agents.energy[a], agents.energy[b]);
  swap(agents.age[a], agents.age[b]);
  swap(agents.alive[a], agents.alive[b]);
  swap(agents.species_id[a], agents.species_id[b]);
  swap(agents.entity_id[a], agents.entity_id[b]);
  for (int i = 0; i < AgentSoA::INPUT_COUNT; ++i) {
    swap(agents.sensors[a * AgentSoA::INPUT_COUNT + i],
         agents.sensors[b * AgentSoA::INPUT_COUNT + i]);
  }
  swap(agents.decision_x[a], agents.decision_x[b]);
  swap(agents.decision_y[a], agents.decision_y[b]);
  swap(agents.distance_traveled[a], agents.distance_traveled[b]);
  swap(agents.offspring_count[a], agents.offspring_count[b]);
}

void swap_specific(PredatorSoA &predator, std::size_t a, std::size_t b) {
  using std::swap;
  swap(predator.kills[a], predator.kills[b]);
}

void swap_specific(PreySoA &prey, std::size_t a, std::size_t b) {
  using std::swap;
  swap(prey.food_eaten[a], prey.food_eaten[b]);
}

void pop_back_agent_fields(PositionSoA &positions, AgentSoA &agents,
                           std::size_t new_size) {
  positions.x.pop_back();
  positions.y.pop_back();
  agents.vel_x.pop_back();
  agents.vel_y.pop_back();
  agents.speed.pop_back();
  agents.energy.pop_back();
  agents.age.pop_back();
  agents.alive.pop_back();
  agents.species_id.pop_back();
  agents.entity_id.pop_back();
  agents.sensors.resize(new_size * AgentSoA::INPUT_COUNT);
  agents.decision_x.pop_back();
  agents.decision_y.pop_back();
  agents.distance_traveled.pop_back();
  agents.offspring_count.pop_back();
}

void pop_back_specific(PredatorSoA &predator) {
  predator.kills.pop_back();
}

void pop_back_specific(PreySoA &prey) {
  prey.food_eaten.pop_back();
}

template <typename RegistryT>
uint32_t find_by_agent_id_impl(const RegistryT &registry, uint32_t agent_id) {
  const auto it = std::find(registry.agents.entity_id.begin(),
                            registry.agents.entity_id.end(), agent_id);
  if (it == registry.agents.entity_id.end()) {
    return INVALID_ENTITY;
  }
  return static_cast<uint32_t>(
      std::distance(registry.agents.entity_id.begin(), it));
}

} // namespace

void FoodStore::initialize(const SimulationConfig &config, Random &rng) {
  resize(static_cast<std::size_t>(config.food_count));
  active.assign(static_cast<std::size_t>(config.food_count), 1);

  const float grid_size = static_cast<float>(config.grid_size);
  for (int i = 0; i < config.food_count; ++i) {
    positions.x[static_cast<std::size_t>(i)] = rng.next_float(0.0f, grid_size);
    positions.y[static_cast<std::size_t>(i)] = rng.next_float(0.0f, grid_size);
  }
}

void FoodStore::respawn_step(const SimulationConfig &config, int step_index,
                             std::uint64_t seed) {
  const float world_size = static_cast<float>(config.grid_size);

  for (std::size_t i = 0; i < active.size(); ++i) {
    if (active[i]) {
      continue;
    }

    const uint32_t slot = static_cast<uint32_t>(i);
    if (!respawn::should_respawn(seed, step_index, slot,
                                 config.food_respawn_rate)) {
      continue;
    }

    positions.x[i] = respawn::respawn_x(seed, step_index, slot, world_size);
    positions.y[i] = respawn::respawn_y(seed, step_index, slot, world_size);
    active[i] = 1;
  }
}

uint32_t PredatorRegistry::create() {
  const uint32_t entity = static_cast<uint32_t>(size());
  resize(size() + 1);
  return entity;
}

void PredatorRegistry::destroy(uint32_t entity) {
  if (valid(entity)) {
    agents.alive[entity] = 0;
  }
}

bool PredatorRegistry::valid(uint32_t entity) const {
  return entity != INVALID_ENTITY && entity < size();
}

std::size_t PredatorRegistry::size() const {
  return registry_size(positions, agents, predator);
}

bool PredatorRegistry::empty() const {
  return size() == 0;
}

void PredatorRegistry::clear() {
  resize(0);
}

RegistryCompactionResult PredatorRegistry::compact_dead() {
  RegistryCompactionResult result;

  std::size_t i = 0;
  while (i < size()) {
    if (agents.alive[i] != 0) {
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

uint32_t PredatorRegistry::find_by_agent_id(uint32_t agent_id) const {
  return find_by_agent_id_impl(*this, agent_id);
}

void PredatorRegistry::resize(std::size_t new_size) {
  resize_registry(positions, agents, predator, new_size);
}

void PredatorRegistry::swap_entities(std::size_t a, std::size_t b) {
  swap_agent_fields(positions, agents, a, b);
  swap_specific(predator, a, b);
}

void PredatorRegistry::pop_back() {
  const std::size_t new_size = size() - 1;
  pop_back_agent_fields(positions, agents, new_size);
  pop_back_specific(predator);
}

uint32_t PreyRegistry::create() {
  const uint32_t entity = static_cast<uint32_t>(size());
  resize(size() + 1);
  return entity;
}

void PreyRegistry::destroy(uint32_t entity) {
  if (valid(entity)) {
    agents.alive[entity] = 0;
  }
}

bool PreyRegistry::valid(uint32_t entity) const {
  return entity != INVALID_ENTITY && entity < size();
}

std::size_t PreyRegistry::size() const {
  return registry_size(positions, agents, prey);
}

bool PreyRegistry::empty() const {
  return size() == 0;
}

void PreyRegistry::clear() {
  resize(0);
}

RegistryCompactionResult PreyRegistry::compact_dead() {
  RegistryCompactionResult result;

  std::size_t i = 0;
  while (i < size()) {
    if (agents.alive[i] != 0) {
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

uint32_t PreyRegistry::find_by_agent_id(uint32_t agent_id) const {
  return find_by_agent_id_impl(*this, agent_id);
}

void PreyRegistry::resize(std::size_t new_size) {
  resize_registry(positions, agents, prey, new_size);
}

void PreyRegistry::swap_entities(std::size_t a, std::size_t b) {
  swap_agent_fields(positions, agents, a, b);
  swap_specific(prey, a, b);
}

void PreyRegistry::pop_back() {
  const std::size_t new_size = size() - 1;
  pop_back_agent_fields(positions, agents, new_size);
  pop_back_specific(prey);
}

} // namespace moonai
