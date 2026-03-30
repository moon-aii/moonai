#pragma once

#include "core/random.hpp"
#include "core/types.hpp"
#include "evolution/genome.hpp"
#include "evolution/mutation.hpp"
#include "evolution/network_cache.hpp"
#include "evolution/species.hpp"
#include "simulation/registry.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace moonai {

struct SimEvent {
  enum Type : uint8_t { Kill, Food, Birth, Death };
  Type type;
  uint32_t agent_id = 0;
  uint32_t target_id = 0;
  Vec2 position;
};

struct PendingOffspring {
  uint32_t parent_a = INVALID_ENTITY;
  uint32_t parent_b = INVALID_ENTITY;
  Vec2 spawn_position;
};

struct EventCounters {
  int kills = 0;
  int food_eaten = 0;
  int births = 0;
  int deaths = 0;

  void clear() {
    kills = 0;
    food_eaten = 0;
    births = 0;
    deaths = 0;
  }

  void add(const EventCounters &other) {
    kills += other.kills;
    food_eaten += other.food_eaten;
    births += other.births;
    deaths += other.deaths;
  }
};

struct LiveMetrics {
  int alive_predators = 0;
  int alive_prey = 0;
  int active_food = 0;
  int predator_species = 0;
  int prey_species = 0;
};

struct ReportMetrics {
  int step = 0;
  int predator_count = 0;
  int prey_count = 0;
  int births = 0;
  int deaths = 0;
  int predator_species = 0;
  int prey_species = 0;
  float avg_genome_complexity = 0.0f;
  float avg_predator_energy = 0.0f;
  float avg_prey_energy = 0.0f;
};

struct RuntimeState {
  explicit RuntimeState(std::uint64_t seed) : rng(seed) {}

  Random rng;
  uint32_t next_agent_id = 1;
  int step = 0;
  std::vector<SimEvent> last_step_events;
  std::vector<PendingOffspring> pending_predator_offspring;
  std::vector<PendingOffspring> pending_prey_offspring;
  EventCounters step_events;
  EventCounters report_events;
  EventCounters total_events;
};

struct PopulationEvolutionState {
  InnovationTracker innovation_tracker;
  std::vector<Species> species;
  std::vector<Genome> genomes;
  NetworkCache network_cache;
};

struct EvolutionState {
  PopulationEvolutionState predators;
  PopulationEvolutionState prey;
};

struct MetricsState {
  LiveMetrics live;
  ReportMetrics last_report;
  std::vector<ReportMetrics> history;
};

struct UiState {
  bool paused = false;
  bool step_requested = false;
  bool reset_requested = false;
  int speed_multiplier = 1;
  uint32_t selected_agent_id = 0;
  std::unordered_map<std::uint32_t, float> selected_node_activations;
};

struct AppState {
  explicit AppState(std::uint64_t seed) : runtime(seed) {}

  PredatorRegistry predators;
  PreyRegistry prey;
  FoodStore food_store;
  EvolutionState evolution;
  RuntimeState runtime;
  MetricsState metrics;
  UiState ui;
};

inline void accumulate_step_events(AppState &state) {
  state.runtime.report_events.add(state.runtime.step_events);
  state.runtime.total_events.add(state.runtime.step_events);
}

inline Genome *predator_genome_for(AppState &state, uint32_t entity) {
  if (entity == INVALID_ENTITY ||
      entity >= state.evolution.predators.genomes.size()) {
    return nullptr;
  }
  return &state.evolution.predators.genomes[entity];
}

inline const Genome *predator_genome_for(const AppState &state,
                                         uint32_t entity) {
  if (entity == INVALID_ENTITY ||
      entity >= state.evolution.predators.genomes.size()) {
    return nullptr;
  }
  return &state.evolution.predators.genomes[entity];
}

inline Genome *prey_genome_for(AppState &state, uint32_t entity) {
  if (entity == INVALID_ENTITY ||
      entity >= state.evolution.prey.genomes.size()) {
    return nullptr;
  }
  return &state.evolution.prey.genomes[entity];
}

inline const Genome *prey_genome_for(const AppState &state, uint32_t entity) {
  if (entity == INVALID_ENTITY ||
      entity >= state.evolution.prey.genomes.size()) {
    return nullptr;
  }
  return &state.evolution.prey.genomes[entity];
}

} // namespace moonai
