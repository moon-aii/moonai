#pragma once

#include "core/types.hpp"
#include "simulation/components.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moonai {

struct Registry {
  struct CompactionResult {
    std::vector<std::pair<uint32_t, uint32_t>> moved;
    std::vector<uint32_t> removed;
  };

  PositionSoA positions;
  MotionSoA motion;
  VitalsSoA vitals;
  IdentitySoA identity;
  SensorSoA sensors;
  StatsSoA stats;
  BrainSoA brain;

  uint32_t create();
  void destroy(uint32_t entity);
  bool valid(uint32_t entity) const;

  size_t size() const {
    return positions.size();
  }

  bool empty() const {
    return size() == 0;
  }

  void clear();
  CompactionResult compact_dead();
  uint32_t find_by_agent_id(uint32_t agent_id) const;
  uint32_t next_agent_id() {
    return next_agent_id_++;
  }

private:
  void resize(std::size_t size);
  void swap_entities(std::size_t a, std::size_t b);
  void pop_back();

  uint32_t next_agent_id_ = 1;
};

} // namespace moonai
