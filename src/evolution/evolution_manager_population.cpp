#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/crossover.hpp"
#include "evolution/mutation.hpp"
#include "gpu/gpu_network_cache.hpp"

#include <algorithm>

namespace moonai {

Genome EvolutionManager::create_initial_genome(AppState &state) const {
  Genome genome(num_inputs_, num_outputs_);
  for (const auto &in_node : genome.nodes()) {
    if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) {
      continue;
    }
    for (const auto &out_node : genome.nodes()) {
      if (out_node.type != NodeType::Output) {
        continue;
      }

      const std::uint32_t innovation =
          state.evolution.innovation_tracker.get_innovation(in_node.id,
                                                            out_node.id);
      genome.add_connection({in_node.id, out_node.id,
                             state.runtime.rng.next_float(-1.0f, 1.0f), true,
                             innovation});
    }
  }
  return genome;
}

Genome EvolutionManager::create_child_genome(AppState &state,
                                             const Genome &parent_a,
                                             const Genome &parent_b) const {
  Genome child = Crossover::crossover(parent_a, parent_b, state.runtime.rng);
  Mutation::mutate(child, state.runtime.rng, config_,
                   state.evolution.innovation_tracker);
  return child;
}

void EvolutionManager::seed_initial_population(AppState &state) {
  state.registry.clear();
  state.evolution.entity_genomes.clear();
  state.evolution.network_cache.clear();
  state.evolution.species.clear();

  const int num_predators = config_.predator_count;
  const int num_prey = config_.prey_count;
  const float grid_size = static_cast<float>(config_.grid_size);

  auto seed_entity = [&](uint8_t type, float speed) {
    const uint32_t idx = state.registry.create();

    Genome genome = create_initial_genome(state);
    if (idx >= state.evolution.entity_genomes.size()) {
      state.evolution.entity_genomes.resize(idx + 1);
    }
    state.evolution.entity_genomes[idx] = genome;
    state.evolution.network_cache.assign(idx,
                                         state.evolution.entity_genomes[idx]);

    state.registry.positions.x[idx] =
        state.runtime.rng.next_float(0.0f, grid_size);
    state.registry.positions.y[idx] =
        state.runtime.rng.next_float(0.0f, grid_size);

    state.registry.motion.vel_x[idx] = 0.0f;
    state.registry.motion.vel_y[idx] = 0.0f;
    state.registry.motion.speed[idx] = speed;

    state.registry.vitals.energy[idx] = config_.initial_energy;
    state.registry.vitals.age[idx] = 0;
    state.registry.vitals.alive[idx] = 1;

    state.registry.identity.type[idx] = type;
    state.registry.identity.species_id[idx] = 0;
    state.registry.identity.entity_id[idx] = state.registry.next_agent_id();

    std::fill(state.registry.sensors.input_ptr(idx),
              state.registry.sensors.input_ptr(idx) + SensorSoA::INPUT_COUNT,
              0.0f);

    state.registry.stats.kills[idx] = 0;
    state.registry.stats.food_eaten[idx] = 0;
    state.registry.stats.distance_traveled[idx] = 0.0f;
    state.registry.stats.offspring_count[idx] = 0;

    state.registry.brain.decision_x[idx] = 0.0f;
    state.registry.brain.decision_y[idx] = 0.0f;
  };

  for (int i = 0; i < num_predators; ++i) {
    seed_entity(IdentitySoA::TYPE_PREDATOR, config_.predator_speed);
  }

  for (int i = 0; i < num_prey; ++i) {
    seed_entity(IdentitySoA::TYPE_PREY, config_.prey_speed);
  }

  if (gpu_network_cache_) {
    gpu_network_cache_->invalidate();
  }
}

uint32_t EvolutionManager::create_offspring(AppState &state, uint32_t parent_a,
                                            uint32_t parent_b,
                                            Vec2 spawn_position) {
  MOONAI_PROFILE_SCOPE("evolution_offspring");
  if (!state.registry.valid(parent_a) || !state.registry.valid(parent_b)) {
    return INVALID_ENTITY;
  }

  if (parent_a >= state.evolution.entity_genomes.size() ||
      parent_b >= state.evolution.entity_genomes.size()) {
    return INVALID_ENTITY;
  }

  const Genome &genome_a = state.evolution.entity_genomes[parent_a];
  const Genome &genome_b = state.evolution.entity_genomes[parent_b];
  Genome child_genome = create_child_genome(state, genome_a, genome_b);

  const uint32_t idx = state.registry.create();

  state.registry.positions.x[idx] = spawn_position.x;
  state.registry.positions.y[idx] = spawn_position.y;
  state.registry.motion.vel_x[idx] = 0.0f;
  state.registry.motion.vel_y[idx] = 0.0f;
  state.registry.motion.speed[idx] = state.registry.motion.speed[parent_a];
  state.registry.vitals.energy[idx] = config_.offspring_initial_energy;
  state.registry.vitals.age[idx] = 0;
  state.registry.vitals.alive[idx] = 1;
  state.registry.identity.type[idx] = state.registry.identity.type[parent_a];
  state.registry.identity.species_id[idx] =
      state.registry.identity.species_id[parent_a];
  state.registry.identity.entity_id[idx] = state.registry.next_agent_id();

  std::fill(state.registry.sensors.input_ptr(idx),
            state.registry.sensors.input_ptr(idx) + SensorSoA::INPUT_COUNT,
            0.0f);

  state.registry.stats.kills[idx] = 0;
  state.registry.stats.food_eaten[idx] = 0;
  state.registry.stats.distance_traveled[idx] = 0.0f;
  state.registry.stats.offspring_count[idx] = 0;

  state.registry.brain.decision_x[idx] = 0.0f;
  state.registry.brain.decision_y[idx] = 0.0f;

  if (idx >= state.evolution.entity_genomes.size()) {
    state.evolution.entity_genomes.resize(idx + 1);
  }
  state.evolution.entity_genomes[idx] = std::move(child_genome);
  state.evolution.network_cache.assign(idx,
                                       state.evolution.entity_genomes[idx]);

  state.registry.vitals.energy[parent_a] -= config_.reproduction_energy_cost;
  state.registry.vitals.energy[parent_b] -= config_.reproduction_energy_cost;

  state.registry.stats.offspring_count[parent_a]++;
  state.registry.stats.offspring_count[parent_b]++;

  if (gpu_network_cache_) {
    gpu_network_cache_->invalidate();
  }

  return idx;
}

void EvolutionManager::refresh_species(AppState &state) {
  auto &species = state.evolution.species;
  for (auto &entry : species) {
    entry.clear_members();
  }
  species.clear();

  const uint32_t entity_count = static_cast<uint32_t>(state.registry.size());
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    if (idx >= state.evolution.entity_genomes.size()) {
      continue;
    }

    Genome &genome = state.evolution.entity_genomes[idx];
    int assigned_species_id = -1;

    for (auto &entry : species) {
      if (entry.is_compatible(genome, config_.compatibility_threshold,
                              config_.c1_excess, config_.c2_disjoint,
                              config_.c3_weight)) {
        entry.add_member(idx, genome);
        assigned_species_id = entry.id();
        break;
      }
    }

    if (assigned_species_id < 0) {
      Species entry(genome);
      entry.add_member(idx, genome);
      assigned_species_id = entry.id();
      species.push_back(std::move(entry));
    }

    state.registry.identity.species_id[idx] =
        static_cast<uint32_t>(assigned_species_id);
  }

  for (auto &entry : species) {
    entry.refresh_summary();
  }
}

void EvolutionManager::on_entity_destroyed(AppState &state, uint32_t entity) {
  if (entity != INVALID_ENTITY && static_cast<std::size_t>(entity) + 1 ==
                                      state.evolution.entity_genomes.size()) {
    state.evolution.entity_genomes.pop_back();
  }
  state.evolution.network_cache.remove(entity);
  if (gpu_network_cache_) {
    gpu_network_cache_->invalidate();
  }
}

void EvolutionManager::on_entity_moved(AppState &state, uint32_t from,
                                       uint32_t to) {
  if (from == to) {
    return;
  }

  if (from < state.evolution.entity_genomes.size()) {
    if (to >= state.evolution.entity_genomes.size()) {
      state.evolution.entity_genomes.resize(static_cast<std::size_t>(to) + 1);
    }
    state.evolution.entity_genomes[to] =
        std::move(state.evolution.entity_genomes[from]);
  }

  state.evolution.network_cache.move_entity(from, to);
  if (gpu_network_cache_) {
    gpu_network_cache_->invalidate();
  }
}

Genome *EvolutionManager::genome_for(AppState &state, uint32_t entity) {
  return moonai::genome_for(state, entity);
}

const Genome *EvolutionManager::genome_for(const AppState &state,
                                           uint32_t entity) const {
  return moonai::genome_for(state, entity);
}

int EvolutionManager::species_count(const AppState &state) const {
  return static_cast<int>(state.evolution.species.size());
}

} // namespace moonai
