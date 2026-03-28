#include "simulation/registry.hpp"

namespace moonai {

Entity Registry::create() {
  uint32_t index;
  uint32_t generation;

  if (!free_slots_.empty()) {
    index = free_slots_.back();
    free_slots_.pop_back();
    generation = generations_[index];
  } else {
    index = next_entity_index_++;
    generation = 1;
    if (index >= generations_.size()) {
      generations_.resize(index + 1, 0);
    }
    generations_[index] = generation;
  }

  Entity e{index, generation};

  size_t dense_idx = sparse_set_.insert(e);
  ensure_capacity(dense_idx + 1);

  return e;
}

void Registry::destroy(Entity e) {
  if (!valid(e)) {
    return;
  }

  sparse_set_.remove(e);

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
  brain_.resize(0);

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
  brain_.resize(required_size);
}

} // namespace moonai
