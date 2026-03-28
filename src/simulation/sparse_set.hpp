#pragma once
#include "simulation/entity.hpp"
#include <cstdint>
#include <limits>
#include <vector>

namespace moonai {

class SparseSet {
public:
  size_t insert(Entity e) {
    if (e.index >= sparse_.size()) {
      sparse_.resize(e.index + 1, invalid_index);
    }

    if (sparse_[e.index] != invalid_index) {
      return sparse_[e.index];
    }

    size_t dense_idx = dense_.size();
    dense_.push_back(e);
    sparse_[e.index] = static_cast<uint32_t>(dense_idx);

    return dense_idx;
  }

  void remove(Entity e) {
    if (!contains(e)) {
      return;
    }

    size_t dense_idx = sparse_[e.index];
    Entity last_entity = dense_.back();

    dense_[dense_idx] = last_entity;
    sparse_[last_entity.index] = static_cast<uint32_t>(dense_idx);

    dense_.pop_back();
    sparse_[e.index] = invalid_index;
  }

  bool contains(Entity e) const {
    if (e.index >= sparse_.size()) {
      return false;
    }
    if (sparse_[e.index] == invalid_index) {
      return false;
    }
    size_t dense_idx = sparse_[e.index];
    return dense_[dense_idx].generation == e.generation;
  }

  size_t get_index(Entity e) const {
    if (!contains(e)) {
      return invalid_index;
    }
    return sparse_[e.index];
  }

  size_t size() const {
    return dense_.size();
  }
  bool empty() const {
    return dense_.empty();
  }

  const std::vector<Entity> &dense() const {
    return dense_;
  }

private:
  static constexpr uint32_t invalid_index =
      std::numeric_limits<uint32_t>::max();

  std::vector<uint32_t> sparse_;
  std::vector<Entity> dense_;
};

} // namespace moonai