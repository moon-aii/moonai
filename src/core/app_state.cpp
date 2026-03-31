#include "core/app_state.hpp"

#include "core/deterministic_respawn.hpp"

#include <algorithm>

namespace moonai {

namespace {

void resize_registry(AgentRegistry &registry, std::size_t new_size) {
  registry.pos_x.resize(new_size);
  registry.pos_y.resize(new_size);
  registry.vel_x.resize(new_size);
  registry.vel_y.resize(new_size);
  registry.energy.resize(new_size);
  registry.age.resize(new_size);
  registry.alive.resize(new_size);
  registry.species_id.resize(new_size);
  registry.entity_id.resize(new_size);
  registry.consumption.resize(new_size);
}

std::size_t registry_size(const AgentRegistry &registry) {
  return registry.pos_x.size();
}

void swap_agent_fields(AgentRegistry &registry, std::size_t a, std::size_t b) {
  using std::swap;

  swap(registry.pos_x[a], registry.pos_x[b]);
  swap(registry.pos_y[a], registry.pos_y[b]);
  swap(registry.vel_x[a], registry.vel_x[b]);
  swap(registry.vel_y[a], registry.vel_y[b]);
  swap(registry.energy[a], registry.energy[b]);
  swap(registry.age[a], registry.age[b]);
  swap(registry.alive[a], registry.alive[b]);
  swap(registry.species_id[a], registry.species_id[b]);
  swap(registry.entity_id[a], registry.entity_id[b]);
  swap(registry.consumption[a], registry.consumption[b]);
}

void pop_back_agent_fields(AgentRegistry &registry, std::size_t new_size) {
  registry.pos_x.pop_back();
  registry.pos_y.pop_back();
  registry.vel_x.pop_back();
  registry.vel_y.pop_back();
  registry.energy.pop_back();
  registry.age.pop_back();
  registry.alive.pop_back();
  registry.species_id.pop_back();
  registry.entity_id.pop_back();
  registry.consumption.pop_back();
}

uint32_t find_by_agent_id_impl(const AgentRegistry &registry,
                               uint32_t agent_id) {
  const auto it =
      std::find(registry.entity_id.begin(), registry.entity_id.end(), agent_id);
  if (it == registry.entity_id.end()) {
    return INVALID_ENTITY;
  }
  return static_cast<uint32_t>(std::distance(registry.entity_id.begin(), it));
}

} // namespace

void Food::initialize(const SimulationConfig &config, Random &rng) {
  const auto count = static_cast<std::size_t>(config.food_count);
  pos_x.resize(count);
  pos_y.resize(count);
  active.resize(count);
  active.assign(count, 1);

  const float grid_size = static_cast<float>(config.grid_size);
  for (int i = 0; i < config.food_count; ++i) {
    pos_x[static_cast<std::size_t>(i)] = rng.next_float(0.0f, grid_size);
    pos_y[static_cast<std::size_t>(i)] = rng.next_float(0.0f, grid_size);
  }
}

void Food::respawn_step(const SimulationConfig &config, int step_index,
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

    pos_x[i] = respawn::respawn_x(seed, step_index, slot, world_size);
    pos_y[i] = respawn::respawn_y(seed, step_index, slot, world_size);
    active[i] = 1;
  }
}

uint32_t AgentRegistry::create() {
  const uint32_t entity = static_cast<uint32_t>(size());
  resize(size() + 1);
  return entity;
}

bool AgentRegistry::valid(uint32_t entity) const {
  return entity != INVALID_ENTITY && entity < size();
}

std::size_t AgentRegistry::size() const {
  return registry_size(*this);
}

void AgentRegistry::clear() {
  resize(0);
}

RegistryCompactionResult AgentRegistry::compact_dead() {
  RegistryCompactionResult result;

  std::size_t i = 0;
  while (i < size()) {
    if (alive[i] != 0) {
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

uint32_t AgentRegistry::find_by_agent_id(uint32_t agent_id) const {
  return find_by_agent_id_impl(*this, agent_id);
}

void AgentRegistry::resize(std::size_t new_size) {
  resize_registry(*this, new_size);
}

void AgentRegistry::swap_entities(std::size_t a, std::size_t b) {
  swap_agent_fields(*this, a, b);
}

void AgentRegistry::pop_back() {
  const std::size_t new_size = size() - 1;
  pop_back_agent_fields(*this, new_size);
}

} // namespace moonai
