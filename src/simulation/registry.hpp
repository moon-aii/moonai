#pragma once
#include "core/types.hpp"
#include "simulation/components.hpp"
#include "simulation/entity.hpp"
#include "simulation/sparse_set.hpp"
#include <cstdint>
#include <optional>
#include <vector>

namespace moonai {

// Registry: Sparse-set ECS with SoA component storage
// Entity handles are stable (never invalidated by other deletions)
// Component arrays are dense and contiguous (cache-friendly)
class Registry {
public:
  // Create entity, returns stable handle
  Entity create();

  // Destroy entity (handle becomes invalid)
  void destroy(Entity e);

  // Check if entity is alive (generation matches)
  bool valid(Entity e) const;
  bool alive(Entity e) const {
    return valid(e);
  }

  // Component access (returns index into dense arrays)
  size_t index_of(Entity e) const {
    return sparse_set_.get_index(e);
  }

  // Number of alive entities
  size_t size() const {
    return sparse_set_.size();
  }
  bool empty() const {
    return sparse_set_.empty();
  }

  // Get component arrays (for systems/GPU packing)
  PositionSoA &positions() {
    return positions_;
  }
  const PositionSoA &positions() const {
    return positions_;
  }

  MotionSoA &motion() {
    return motion_;
  }
  const MotionSoA &motion() const {
    return motion_;
  }

  VitalsSoA &vitals() {
    return vitals_;
  }
  const VitalsSoA &vitals() const {
    return vitals_;
  }

  IdentitySoA &identity() {
    return identity_;
  }
  const IdentitySoA &identity() const {
    return identity_;
  }

  SensorSoA &sensors() {
    return sensors_;
  }
  const SensorSoA &sensors() const {
    return sensors_;
  }

  StatsSoA &stats() {
    return stats_;
  }
  const StatsSoA &stats() const {
    return stats_;
  }

  VisualSoA &visual() {
    return visual_;
  }
  const VisualSoA &visual() const {
    return visual_;
  }

  BrainSoA &brain() {
    return brain_;
  }
  const BrainSoA &brain() const {
    return brain_;
  }

  FoodStateSoA &food_state() {
    return food_state_;
  }
  const FoodStateSoA &food_state() const {
    return food_state_;
  }

  // Create food entity (recyclable slot-based)
  Entity create_food(Vec2 position, uint32_t slot_index, float radius,
                     uint32_t color_rgba);

  // Query all food entities
  std::vector<Entity> query_food() const;

  // Check if entity is food
  static bool is_food(const IdentitySoA &identity, size_t idx) {
    return identity.type[idx] == IdentitySoA::TYPE_FOOD;
  }

  // Direct component access by entity
  float &pos_x(Entity e) {
    return positions_.x[index_of(e)];
  }
  float &pos_y(Entity e) {
    return positions_.y[index_of(e)];
  }
  float &energy(Entity e) {
    return vitals_.energy[index_of(e)];
  }

  // Remove all entities
  void clear();

  // Get list of all living entities (for GPU packing)
  const std::vector<Entity> &living_entities() const {
    return sparse_set_.dense();
  }

  // Get sparse set for advanced queries
  const SparseSet &sparse_set() const {
    return sparse_set_;
  }

private:
  void ensure_capacity(size_t required_size);

  SparseSet sparse_set_;

  // Component storage (dense arrays, indexed by sparse_set)
  PositionSoA positions_;
  MotionSoA motion_;
  VitalsSoA vitals_;
  IdentitySoA identity_;
  SensorSoA sensors_;
  StatsSoA stats_;
  VisualSoA visual_;
  BrainSoA brain_;
  FoodStateSoA food_state_;

  // Entity slot recycling
  uint32_t next_entity_index_ = 1;
  std::vector<uint32_t> free_slots_;
  std::vector<uint32_t> generations_;
};

} // namespace moonai