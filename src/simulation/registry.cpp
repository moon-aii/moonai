#include "simulation/registry.hpp"
#include <algorithm>

namespace moonai {

Entity Registry::create() {
  uint32_t index;
  uint32_t generation;

  if (!free_slots_.empty()) {
    // Recycle a slot
    index = free_slots_.back();
    free_slots_.pop_back();
    generation = generations_[index];
  } else {
    // Create new slot
    index = next_entity_index_++;
    generation = 1;
    if (index >= generations_.size()) {
      generations_.resize(index + 1, 0);
    }
    generations_[index] = generation;
  }

  Entity e{index, generation};

  // Insert into sparse set and resize component arrays
  size_t dense_idx = sparse_set_.insert(e);
  ensure_capacity(dense_idx + 1);

  return e;
}

void Registry::destroy(Entity e) {
  if (!valid(e)) {
    return;
  }

  // Remove from sparse set (this doesn't invalidate other handles)
  sparse_set_.remove(e);

  // Mark slot for recycling
  generations_[e.index]++;
  free_slots_.push_back(e.index);
}

bool Registry::valid(Entity e) const {
  if (e.index == 0 || e.index >= generations_.size()) {
    return false;
  }
  return generations_[e.index] == e.generation && sparse_set_.contains(e);
}

void Registry::clear() {
  sparse_set_ = SparseSet{};

  positions_.resize(0);
  motion_.resize(0);
  vitals_.resize(0);
  identity_.resize(0);
  sensors_.resize(0);
  stats_.resize(0);
  visual_.resize(0);
  brain_.resize(0);
  food_state_.resize(0);

  next_entity_index_ = 1;
  free_slots_.clear();
  generations_.clear();
}

void Registry::ensure_capacity(size_t required_size) {
  if (required_size <= positions_.size()) {
    return;
  }

  positions_.resize(required_size);
  motion_.resize(required_size);
  vitals_.resize(required_size);
  identity_.resize(required_size);
  sensors_.resize(required_size);
  stats_.resize(required_size);
  visual_.resize(required_size);
  brain_.resize(required_size);
  food_state_.resize(required_size);
}

Entity Registry::create_food(Vec2 position, uint32_t slot_index, float radius,
                             uint32_t color_rgba) {
  Entity e = create();
  size_t idx = index_of(e);

  // Identity
  identity_.type[idx] = IdentitySoA::TYPE_FOOD;
  identity_.species_id[idx] = 0; // Food has no species
  identity_.entity_id[idx] = e.index;

  // Position
  positions_.x[idx] = position.x;
  positions_.y[idx] = position.y;

  // Vitals
  vitals_.energy[idx] = 0.0f; // Food has no energy
  vitals_.age[idx] = 0;       // Food doesn't age
  vitals_.alive[idx] = 1;
  vitals_.reproduction_cooldown[idx] = 0;

  // Visual
  visual_.radius[idx] = radius;
  visual_.color_rgba[idx] = color_rgba;
  visual_.shape_type[idx] = 0; // Circle

  // Food state
  food_state_.slot_index[idx] = slot_index;
  food_state_.active[idx] = 1;

  // Stats
  stats_.kills[idx] = 0;
  stats_.food_eaten[idx] = 0;
  stats_.distance_traveled[idx] = 0.0f;
  stats_.offspring_count[idx] = 0;

  // Motion
  motion_.vel_x[idx] = 0.0f;
  motion_.vel_y[idx] = 0.0f;
  motion_.speed[idx] = 0.0f;

  return e;
}

std::vector<Entity> Registry::query_food() const {
  std::vector<Entity> food_entities;
  const auto &living = living_entities();

  for (Entity e : living) {
    size_t idx = index_of(e);
    if (identity_.type[idx] == IdentitySoA::TYPE_FOOD &&
        food_state_.active[idx]) {
      food_entities.push_back(e);
    }
  }

  return food_entities;
}

} // namespace moonai