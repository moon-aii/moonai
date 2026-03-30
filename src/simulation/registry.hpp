#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"
#include "simulation/components.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace moonai {

struct RegistryCompactionResult {
  std::vector<std::pair<uint32_t, uint32_t>> moved;
  std::vector<uint32_t> removed;
};

struct FoodStore {
  void initialize(const SimulationConfig &config, Random &rng);
  void respawn_step(const SimulationConfig &config, int step_index,
                    std::uint64_t seed);

  void resize(std::size_t n) {
    positions.resize(n);
    active.resize(n);
  }

  std::size_t size() const {
    return positions.size();
  }

  PositionSoA positions;
  std::vector<uint8_t> active;
};

struct PredatorRegistry {
  PositionSoA positions;
  AgentSoA agents;
  PredatorSoA predator;

  uint32_t create();
  void destroy(uint32_t entity);
  bool valid(uint32_t entity) const;
  std::size_t size() const;
  bool empty() const;

  void clear();
  RegistryCompactionResult compact_dead();
  uint32_t find_by_agent_id(uint32_t agent_id) const;

private:
  void resize(std::size_t size);
  void swap_entities(std::size_t a, std::size_t b);
  void pop_back();
};

struct PreyRegistry {
  PositionSoA positions;
  AgentSoA agents;
  PreySoA prey;

  uint32_t create();
  void destroy(uint32_t entity);
  bool valid(uint32_t entity) const;
  std::size_t size() const;
  bool empty() const;

  void clear();
  RegistryCompactionResult compact_dead();
  uint32_t find_by_agent_id(uint32_t agent_id) const;

private:
  void resize(std::size_t size);
  void swap_entities(std::size_t a, std::size_t b);
  void pop_back();
};

} // namespace moonai
