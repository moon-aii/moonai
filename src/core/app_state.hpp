#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "evolution/genome.hpp"
#include "evolution/inference_cache.hpp"
#include "evolution/mutation.hpp"
#include "evolution/network_cache.hpp"
#include "evolution/species.hpp"
#include "simulation/batch.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moonai {

struct UiState {
  bool paused = false;
  bool step_requested = false;
  int speed_multiplier = 1;
  uint32_t selected_agent_id = 0;
};

struct Food {
  void initialize(const SimulationConfig &config, Random &rng);
  void respawn_step(const SimulationConfig &config, int step_index, std::uint64_t seed);

  std::size_t size() const {
    return pos_x.size();
  }

  std::vector<float> pos_x;
  std::vector<float> pos_y;
  std::vector<uint8_t> active;
};

struct AgentRegistry {
  std::vector<float> pos_x;
  std::vector<float> pos_y;
  std::vector<float> vel_x;
  std::vector<float> vel_y;
  std::vector<float> energy;
  std::vector<int> age;
  std::vector<uint8_t> alive;
  std::vector<uint32_t> species_id;
  std::vector<uint32_t> entity_id;
  std::vector<int> generation;

  InnovationTracker innovation_tracker;
  std::vector<Species> species;
  std::vector<Genome> genomes;
  NetworkCache network_cache;
  evolution::InferenceCache inference_cache;

  uint32_t create();
  bool valid(uint32_t entity) const;
  std::size_t size() const;
  void clear();
  void compact();
  uint32_t find_by_agent_id(uint32_t agent_id) const;

private:
  void resize(std::size_t size);
  void swap_entities(std::size_t a, std::size_t b);
  void pop_back();
};

struct MetricsSnapshot {
  int step = 0;
  int predator_count = 0;
  int prey_count = 0;
  int active_food = 0;
  int kills = 0;
  int food_eaten = 0;
  int predator_births = 0;
  int prey_births = 0;
  int predator_deaths = 0;
  int prey_deaths = 0;
  int predator_species = 0;
  int prey_species = 0;
  float avg_predator_complexity = 0.0f;
  float avg_prey_complexity = 0.0f;
  float avg_predator_energy = 0.0f;
  float avg_prey_energy = 0.0f;
  int max_predator_generation = 0;
  float avg_predator_generation = 0.0f;
  int max_prey_generation = 0;
  float avg_prey_generation = 0.0f;
};

struct RuntimeState {
  explicit RuntimeState(int seed) : rng(seed) {}

  Random rng;
  uint32_t next_agent_id = 1;
  int step = 0;
};

struct StepBuffers {
  std::vector<uint8_t> was_food_active;
  std::vector<float> predator_sensors;
  std::vector<float> prey_sensors;
  std::vector<float> predator_decisions;
  std::vector<float> prey_decisions;
  std::vector<int> food_consumed_by;
  std::vector<int> killed_by;
  std::vector<uint32_t> kill_counts;
};

struct AppState {
  explicit AppState(int seed) : runtime(seed) {}

  UiState ui;

  AgentRegistry predator;
  AgentRegistry prey;
  Food food;

  MetricsSnapshot metrics;
  RuntimeState runtime;
  StepBuffers step_buffers;
  simulation::Batch batch;
};

} // namespace moonai
