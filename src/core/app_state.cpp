#include "core/app_state.hpp"
#include "core/deterministic_respawn.hpp"
#include "core/profiler_macros.hpp"

#include <algorithm>

namespace moonai {

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

void Food::respawn_step(const SimulationConfig &config, int step_index, std::uint64_t seed) {
  MOONAI_PROFILE_SCOPE("food_respawn");

  const float world_size = static_cast<float>(config.grid_size);

  for (std::size_t i = 0; i < active.size(); ++i) {
    if (active[i]) {
      continue;
    }

    const uint32_t slot = static_cast<uint32_t>(i);
    if (!respawn::should_respawn(seed, step_index, slot, config.food_respawn_rate)) {
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
  return this->pos_x.size();
}

void AgentRegistry::clear() {
  resize(0);
}

void AgentRegistry::compact() {
  std::size_t i = 0;
  while (i < size()) {
    if (alive[i] != 0) {
      ++i;
      continue;
    }

    const std::size_t last = size() - 1;
    if (i != last) {
      swap_entities(i, last);

      if (last < genomes.size()) {
        if (i >= genomes.size()) {
          genomes.resize(static_cast<std::size_t>(i) + 1);
        }
        genomes[i] = std::move(genomes[last]);
      }

      network_cache.move_entity(static_cast<uint32_t>(last), static_cast<uint32_t>(i));
      inference_cache.swap_remove_entity(static_cast<uint32_t>(i), static_cast<uint32_t>(last));

      if (last + 1 == genomes.size()) {
        genomes.pop_back();
      }
      network_cache.remove(static_cast<uint32_t>(last));
    } else {
      if (!genomes.empty() && last < genomes.size()) {
        genomes.pop_back();
      }
      inference_cache.swap_remove_entity(static_cast<uint32_t>(last), static_cast<uint32_t>(last));
      network_cache.remove(static_cast<uint32_t>(last));
    }

    pop_back();
  }
}

uint32_t AgentRegistry::find_by_agent_id(uint32_t agent_id) const {
  const auto it = std::find(this->entity_id.begin(), this->entity_id.end(), agent_id);
  if (it == this->entity_id.end()) {
    return INVALID_ENTITY;
  }
  return static_cast<uint32_t>(std::distance(this->entity_id.begin(), it));
}

void AgentRegistry::resize(std::size_t new_size) {
  this->pos_x.resize(new_size);
  this->pos_y.resize(new_size);
  this->vel_x.resize(new_size);
  this->vel_y.resize(new_size);
  this->energy.resize(new_size);
  this->age.resize(new_size);
  this->alive.resize(new_size);
  this->species_id.resize(new_size);
  this->entity_id.resize(new_size);
  this->generation.resize(new_size);
}

void AgentRegistry::swap_entities(std::size_t a, std::size_t b) {
  using std::swap;

  swap(this->pos_x[a], this->pos_x[b]);
  swap(this->pos_y[a], this->pos_y[b]);
  swap(this->vel_x[a], this->vel_x[b]);
  swap(this->vel_y[a], this->vel_y[b]);
  swap(this->energy[a], this->energy[b]);
  swap(this->age[a], this->age[b]);
  swap(this->alive[a], this->alive[b]);
  swap(this->species_id[a], this->species_id[b]);
  swap(this->entity_id[a], this->entity_id[b]);
  swap(this->generation[a], this->generation[b]);
}

void AgentRegistry::pop_back() {
  this->pos_x.pop_back();
  this->pos_y.pop_back();
  this->vel_x.pop_back();
  this->vel_y.pop_back();
  this->energy.pop_back();
  this->age.pop_back();
  this->alive.pop_back();
  this->species_id.pop_back();
  this->entity_id.pop_back();
  this->generation.pop_back();
}

} // namespace moonai
