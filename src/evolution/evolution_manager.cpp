#include "evolution/evolution_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/crossover.hpp"
#include "gpu/gpu_batch.hpp"
#include "gpu/gpu_network_cache.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace moonai {

EvolutionManager::EvolutionManager(const SimulationConfig &config)
    : config_(config) {}

EvolutionManager::~EvolutionManager() = default;

void EvolutionManager::initialize(AppState &state, int num_inputs,
                                  int num_outputs) {
  num_inputs_ = num_inputs;
  num_outputs_ = num_outputs;

  state.evolution.innovation_tracker = InnovationTracker();
  state.evolution.innovation_tracker.set_counters(
      0, static_cast<std::uint32_t>(num_inputs_ + num_outputs_ + 1));
  state.evolution.species.clear();
  state.evolution.entity_genomes.clear();
  state.evolution.network_cache.clear();

  if (gpu_network_cache_) {
    gpu_network_cache_->invalidate();
  }
}

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
    const Entity entity = state.registry.create();
    const std::size_t idx = state.registry.index_of(entity);

    Genome genome = create_initial_genome(state);
    if (idx >= state.evolution.entity_genomes.size()) {
      state.evolution.entity_genomes.resize(idx + 1);
    }
    state.evolution.entity_genomes[idx] = genome;
    state.evolution.network_cache.assign(entity,
                                         state.evolution.entity_genomes[idx]);

    state.registry.positions().x[idx] =
        state.runtime.rng.next_float(0.0f, grid_size);
    state.registry.positions().y[idx] =
        state.runtime.rng.next_float(0.0f, grid_size);

    state.registry.motion().vel_x[idx] = 0.0f;
    state.registry.motion().vel_y[idx] = 0.0f;
    state.registry.motion().speed[idx] = speed;

    state.registry.vitals().energy[idx] = config_.initial_energy;
    state.registry.vitals().age[idx] = 0;
    state.registry.vitals().alive[idx] = 1;

    state.registry.identity().type[idx] = type;
    state.registry.identity().species_id[idx] = 0;
    state.registry.identity().entity_id[idx] = state.registry.next_agent_id();

    std::fill(state.registry.sensors().input_ptr(idx),
              state.registry.sensors().input_ptr(idx) + SensorSoA::INPUT_COUNT,
              0.0f);

    state.registry.stats().kills[idx] = 0;
    state.registry.stats().food_eaten[idx] = 0;
    state.registry.stats().distance_traveled[idx] = 0.0f;
    state.registry.stats().offspring_count[idx] = 0;

    state.registry.brain().decision_x[idx] = 0.0f;
    state.registry.brain().decision_y[idx] = 0.0f;
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

Entity EvolutionManager::create_offspring(AppState &state, Entity parent_a,
                                          Entity parent_b,
                                          Vec2 spawn_position) {
  MOONAI_PROFILE_SCOPE("evolution_offspring");
  if (!state.registry.valid(parent_a) || !state.registry.valid(parent_b)) {
    return INVALID_ENTITY;
  }

  if (parent_a.index >= state.evolution.entity_genomes.size() ||
      parent_b.index >= state.evolution.entity_genomes.size()) {
    return INVALID_ENTITY;
  }

  const Genome &genome_a = state.evolution.entity_genomes[parent_a.index];
  const Genome &genome_b = state.evolution.entity_genomes[parent_b.index];
  Genome child_genome = create_child_genome(state, genome_a, genome_b);

  const Entity child = state.registry.create();
  const std::size_t idx = state.registry.index_of(child);
  const std::size_t parent_idx = state.registry.index_of(parent_a);

  state.registry.positions().x[idx] = spawn_position.x;
  state.registry.positions().y[idx] = spawn_position.y;
  state.registry.motion().vel_x[idx] = 0.0f;
  state.registry.motion().vel_y[idx] = 0.0f;
  state.registry.motion().speed[idx] =
      state.registry.motion().speed[parent_idx];
  state.registry.vitals().energy[idx] = config_.offspring_initial_energy;
  state.registry.vitals().age[idx] = 0;
  state.registry.vitals().alive[idx] = 1;
  state.registry.identity().type[idx] =
      state.registry.identity().type[parent_idx];
  state.registry.identity().species_id[idx] =
      state.registry.identity().species_id[parent_idx];
  state.registry.identity().entity_id[idx] = state.registry.next_agent_id();

  std::fill(state.registry.sensors().input_ptr(idx),
            state.registry.sensors().input_ptr(idx) + SensorSoA::INPUT_COUNT,
            0.0f);

  state.registry.stats().kills[idx] = 0;
  state.registry.stats().food_eaten[idx] = 0;
  state.registry.stats().distance_traveled[idx] = 0.0f;
  state.registry.stats().offspring_count[idx] = 0;

  state.registry.brain().decision_x[idx] = 0.0f;
  state.registry.brain().decision_y[idx] = 0.0f;

  if (idx >= state.evolution.entity_genomes.size()) {
    state.evolution.entity_genomes.resize(idx + 1);
  }
  state.evolution.entity_genomes[idx] = std::move(child_genome);
  state.evolution.network_cache.assign(child,
                                       state.evolution.entity_genomes[idx]);

  state.registry.vitals().energy[state.registry.index_of(parent_a)] -=
      config_.reproduction_energy_cost;
  state.registry.vitals().energy[state.registry.index_of(parent_b)] -=
      config_.reproduction_energy_cost;

  state.registry.stats().offspring_count[state.registry.index_of(parent_a)]++;
  state.registry.stats().offspring_count[state.registry.index_of(parent_b)]++;

  if (gpu_network_cache_) {
    gpu_network_cache_->invalidate();
  }

  return child;
}

void EvolutionManager::refresh_species(AppState &state) {
  auto &species = state.evolution.species;
  for (auto &entry : species) {
    entry.clear_members();
  }
  species.clear();

  for (std::size_t idx = 0; idx < state.registry.size(); ++idx) {
    if (idx >= state.evolution.entity_genomes.size()) {
      continue;
    }

    const Entity entity{static_cast<uint32_t>(idx)};
    Genome &genome = state.evolution.entity_genomes[idx];
    int assigned_species_id = -1;

    for (auto &entry : species) {
      if (entry.is_compatible(genome, config_.compatibility_threshold,
                              config_.c1_excess, config_.c2_disjoint,
                              config_.c3_weight)) {
        entry.add_member(entity, genome);
        assigned_species_id = entry.id();
        break;
      }
    }

    if (assigned_species_id < 0) {
      Species entry(genome);
      entry.add_member(entity, genome);
      assigned_species_id = entry.id();
      species.push_back(std::move(entry));
    }

    state.registry.identity().species_id[idx] =
        static_cast<uint32_t>(assigned_species_id);
  }

  for (auto &entry : species) {
    entry.refresh_summary();
  }
}

void EvolutionManager::compute_actions(AppState &state) {
  MOONAI_PROFILE_SCOPE("evolution_compute_actions");
  const std::size_t entity_count = state.registry.size();

  std::vector<float> all_inputs;
  all_inputs.reserve(entity_count * SensorSoA::INPUT_COUNT);

  for (std::size_t idx = 0; idx < entity_count; ++idx) {
    const float *input_ptr = state.registry.sensors().input_ptr(idx);
    all_inputs.insert(all_inputs.end(), input_ptr,
                      input_ptr + SensorSoA::INPUT_COUNT);
  }

  std::vector<float> all_outputs;
  compute_actions_batch(entity_count, state, all_inputs, all_outputs);

  for (std::size_t i = 0; i < entity_count; ++i) {
    state.registry.brain().decision_x[i] = all_outputs[i * 2];
    state.registry.brain().decision_y[i] = all_outputs[i * 2 + 1];
  }
}

void EvolutionManager::compute_actions_batch(
    std::size_t entity_count, AppState &state,
    const std::vector<float> &all_inputs, std::vector<float> &all_outputs) {
  state.evolution.network_cache.activate_batch(
      entity_count, all_inputs, all_outputs, SensorSoA::INPUT_COUNT,
      SensorSoA::OUTPUT_COUNT);
}

void EvolutionManager::on_entity_destroyed(AppState &state, Entity entity) {
  if (entity != INVALID_ENTITY &&
      entity.index + 1 == state.evolution.entity_genomes.size()) {
    state.evolution.entity_genomes.pop_back();
  }
  state.evolution.network_cache.remove(entity);
  if (gpu_network_cache_) {
    gpu_network_cache_->invalidate();
  }
}

void EvolutionManager::on_entity_moved(AppState &state, Entity from,
                                       Entity to) {
  if (from == to) {
    return;
  }

  if (from.index < state.evolution.entity_genomes.size()) {
    if (to.index >= state.evolution.entity_genomes.size()) {
      state.evolution.entity_genomes.resize(to.index + 1);
    }
    state.evolution.entity_genomes[to.index] =
        std::move(state.evolution.entity_genomes[from.index]);
  }

  state.evolution.network_cache.move_entity(from, to);
  if (gpu_network_cache_) {
    gpu_network_cache_->invalidate();
  }
}

Genome *EvolutionManager::genome_for(AppState &state, Entity entity) {
  return moonai::genome_for(state, entity);
}

const Genome *EvolutionManager::genome_for(const AppState &state,
                                           Entity entity) const {
  return moonai::genome_for(state, entity);
}

int EvolutionManager::species_count(const AppState &state) const {
  return static_cast<int>(state.evolution.species.size());
}

void EvolutionManager::enable_gpu(bool use_gpu) {
  use_gpu_ = use_gpu;
  if (use_gpu_ && !gpu_network_cache_) {
    gpu_network_cache_ = std::make_unique<gpu::GpuNetworkCache>();
    gpu_network_cache_->invalidate();
    spdlog::info("GPU neural inference enabled");
  } else if (!use_gpu_) {
    gpu_network_cache_.reset();
    spdlog::info("GPU neural inference disabled");
  }
}

bool EvolutionManager::launch_gpu_neural(AppState &state,
                                         gpu::GpuBatch &gpu_batch,
                                         std::size_t agent_count) {
  MOONAI_PROFILE_SCOPE("gpu_neural", gpu_batch.stream());

  if (!gpu_network_cache_) {
    spdlog::error("GPU neural cache not initialized");
    return false;
  }

  std::vector<std::pair<Entity, int>> network_entities_with_indices;
  {
    MOONAI_PROFILE_SCOPE("gpu_network_scan");
    network_entities_with_indices.reserve(agent_count);

    for (std::size_t gpu_idx = 0; gpu_idx < agent_count; ++gpu_idx) {
      const Entity entity{static_cast<uint32_t>(gpu_idx)};
      if (entity != INVALID_ENTITY &&
          state.evolution.network_cache.has(entity)) {
        network_entities_with_indices.emplace_back(entity,
                                                   static_cast<int>(gpu_idx));
      }
    }

    if (network_entities_with_indices.empty()) {
      spdlog::warn("No entities with neural networks found in GPU batch");
      return true;
    }
  }

  if (gpu_network_cache_->is_dirty() ||
      gpu_network_cache_->entity_mapping().size() !=
          network_entities_with_indices.size() ||
      !std::equal(
          gpu_network_cache_->entity_mapping().begin(),
          gpu_network_cache_->entity_mapping().end(),
          network_entities_with_indices.begin(),
          network_entities_with_indices.end(),
          [](Entity entity, const std::pair<Entity, int> &entity_with_index) {
            return entity == entity_with_index.first;
          })) {
    MOONAI_PROFILE_SCOPE("gpu_cache_build");
    spdlog::debug("Rebuilding GPU network cache for {} network entities",
                  network_entities_with_indices.size());
    gpu_network_cache_->build_from(state.evolution.network_cache,
                                   network_entities_with_indices);
  }

  if (!gpu_network_cache_->launch_inference_async(
          gpu_batch.buffer().device_agent_sensor_inputs(),
          gpu_batch.buffer().device_agent_brain_outputs(),
          network_entities_with_indices.size(), gpu_batch.stream())) {
    gpu_batch.mark_error();
    return false;
  }

  return true;
}

} // namespace moonai
