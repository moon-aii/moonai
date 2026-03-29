#include "evolution/evolution_manager.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/crossover.hpp"
#include "gpu/gpu_batch.hpp"
#include "gpu/gpu_network_cache.hpp"
#include "simulation/registry.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace moonai {

EvolutionManager::EvolutionManager(const SimulationConfig &config, Random &rng)
    : config_(config), rng_(rng) {}

EvolutionManager::~EvolutionManager() = default;

void EvolutionManager::initialize(int num_inputs, int num_outputs) {
  num_inputs_ = num_inputs;
  num_outputs_ = num_outputs;
  species_.clear();
}

Genome EvolutionManager::create_initial_genome() const {
  Genome genome(num_inputs_, num_outputs_);
  for (const auto &in_node : genome.nodes()) {
    if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) {
      continue;
    }
    for (const auto &out_node : genome.nodes()) {
      if (out_node.type != NodeType::Output) {
        continue;
      }
      const std::uint32_t innov =
          const_cast<InnovationTracker &>(tracker_).get_innovation(in_node.id,
                                                                   out_node.id);
      genome.add_connection({in_node.id, out_node.id,
                             const_cast<Random &>(rng_).next_float(-1.0f, 1.0f),
                             true, innov});
    }
  }
  return genome;
}

Genome EvolutionManager::create_child_genome(const Genome &parent_a,
                                             const Genome &parent_b) const {
  Genome child = Crossover::crossover(parent_a, parent_b, rng_);
  Mutation::mutate(child, rng_, config_,
                   const_cast<InnovationTracker &>(tracker_));
  return child;
}

void EvolutionManager::seed_initial_population(Registry &registry) {
  entity_genomes_.clear();
  network_cache_.clear();

  int num_predators = config_.predator_count;
  int num_prey = config_.prey_count;

  float grid_size_f = static_cast<float>(config_.grid_size);

  for (int i = 0; i < num_predators; ++i) {
    Entity e = registry.create();
    size_t idx = registry.index_of(e);

    Genome genome = create_initial_genome();
    entity_genomes_[e] = genome;
    network_cache_.assign(e, genome, config_.activation_function);

    registry.positions().x[idx] = rng_.next_float(0.0f, grid_size_f);
    registry.positions().y[idx] = rng_.next_float(0.0f, grid_size_f);

    registry.motion().vel_x[idx] = 0.0f;
    registry.motion().vel_y[idx] = 0.0f;
    registry.motion().speed[idx] = config_.predator_speed;

    registry.vitals().energy[idx] = config_.initial_energy;
    registry.vitals().age[idx] = 0;
    registry.vitals().alive[idx] = 1;
    registry.vitals().reproduction_cooldown[idx] = 0;

    registry.identity().type[idx] = IdentitySoA::TYPE_PREDATOR;
    registry.identity().species_id[idx] = 0;
    registry.identity().entity_id[idx] = e.index;

    std::fill(registry.sensors().input_ptr(idx),
              registry.sensors().input_ptr(idx) + SensorSoA::INPUT_COUNT, 0.0f);
    std::fill(registry.sensors().output_ptr(idx),
              registry.sensors().output_ptr(idx) + SensorSoA::OUTPUT_COUNT,
              0.0f);

    registry.stats().kills[idx] = 0;
    registry.stats().food_eaten[idx] = 0;
    registry.stats().distance_traveled[idx] = 0.0f;
    registry.stats().offspring_count[idx] = 0;

    registry.brain().decision_x[idx] = 0.0f;
    registry.brain().decision_y[idx] = 0.0f;
  }

  for (int i = 0; i < num_prey; ++i) {
    Entity e = registry.create();
    size_t idx = registry.index_of(e);

    Genome genome = create_initial_genome();
    entity_genomes_[e] = genome;
    network_cache_.assign(e, genome, config_.activation_function);

    registry.positions().x[idx] = rng_.next_float(0.0f, grid_size_f);
    registry.positions().y[idx] = rng_.next_float(0.0f, grid_size_f);

    registry.motion().vel_x[idx] = 0.0f;
    registry.motion().vel_y[idx] = 0.0f;
    registry.motion().speed[idx] = config_.prey_speed;

    registry.vitals().energy[idx] = config_.initial_energy;
    registry.vitals().age[idx] = 0;
    registry.vitals().alive[idx] = 1;
    registry.vitals().reproduction_cooldown[idx] = 0;

    registry.identity().type[idx] = IdentitySoA::TYPE_PREY;
    registry.identity().species_id[idx] = 0;
    registry.identity().entity_id[idx] = e.index;

    std::fill(registry.sensors().input_ptr(idx),
              registry.sensors().input_ptr(idx) + SensorSoA::INPUT_COUNT, 0.0f);
    std::fill(registry.sensors().output_ptr(idx),
              registry.sensors().output_ptr(idx) + SensorSoA::OUTPUT_COUNT,
              0.0f);

    registry.stats().kills[idx] = 0;
    registry.stats().food_eaten[idx] = 0;
    registry.stats().distance_traveled[idx] = 0.0f;
    registry.stats().offspring_count[idx] = 0;

    registry.brain().decision_x[idx] = 0.0f;
    registry.brain().decision_y[idx] = 0.0f;
  }
}

Entity EvolutionManager::create_offspring(Registry &registry, Entity parent_a,
                                          Entity parent_b,
                                          Vec2 spawn_position) {
  MOONAI_PROFILE_SCOPE("evolution_offspring");
  if (!registry.valid(parent_a) || !registry.valid(parent_b)) {
    return INVALID_ENTITY;
  }

  auto it_a = entity_genomes_.find(parent_a);
  auto it_b = entity_genomes_.find(parent_b);
  if (it_a == entity_genomes_.end() || it_b == entity_genomes_.end()) {
    return INVALID_ENTITY;
  }

  const Genome &genome_a = it_a->second;
  const Genome &genome_b = it_b->second;

  Genome child_genome = create_child_genome(genome_a, genome_b);

  Entity child = registry.create();

  size_t idx = registry.index_of(child);
  size_t parent_idx = registry.index_of(parent_a);

  registry.positions().x[idx] = spawn_position.x;
  registry.positions().y[idx] = spawn_position.y;
  registry.motion().vel_x[idx] = 0.0f;
  registry.motion().vel_y[idx] = 0.0f;
  registry.motion().speed[idx] = registry.motion().speed[parent_idx];
  registry.vitals().energy[idx] = config_.offspring_initial_energy;
  registry.vitals().age[idx] = 0;
  registry.vitals().alive[idx] = 1;
  registry.vitals().reproduction_cooldown[idx] = 0;
  registry.identity().type[idx] = registry.identity().type[parent_idx];
  registry.identity().species_id[idx] =
      registry.identity().species_id[parent_idx];
  registry.identity().entity_id[idx] = child.index;

  std::fill(registry.sensors().input_ptr(idx),
            registry.sensors().input_ptr(idx) + SensorSoA::INPUT_COUNT, 0.0f);
  std::fill(registry.sensors().output_ptr(idx),
            registry.sensors().output_ptr(idx) + SensorSoA::OUTPUT_COUNT, 0.0f);

  registry.stats().kills[idx] = 0;
  registry.stats().food_eaten[idx] = 0;
  registry.stats().distance_traveled[idx] = 0.0f;
  registry.stats().offspring_count[idx] = 0;

  registry.brain().decision_x[idx] = 0.0f;
  registry.brain().decision_y[idx] = 0.0f;

  entity_genomes_[child] = std::move(child_genome);

  network_cache_.assign(child, entity_genomes_[child],
                        config_.activation_function);

  registry.vitals().energy[registry.index_of(parent_a)] -=
      config_.reproduction_energy_cost;
  registry.vitals().energy[registry.index_of(parent_b)] -=
      config_.reproduction_energy_cost;

  registry.stats().offspring_count[registry.index_of(parent_a)]++;
  registry.stats().offspring_count[registry.index_of(parent_b)]++;

  return child;
}

void EvolutionManager::refresh_species(Registry &registry) {
  for (auto &species : species_) {
    species.clear_members();
  }
  species_.clear();

  for (Entity e : registry.living_entities()) {
    auto it = entity_genomes_.find(e);
    if (it == entity_genomes_.end())
      continue;

    Genome &genome = it->second;
    bool assigned = false;

    for (auto &species : species_) {
      if (species.is_compatible(genome, config_.compatibility_threshold,
                                config_.c1_excess, config_.c2_disjoint,
                                config_.c3_weight)) {
        species.add_member(e, genome);
        assigned = true;
        break;
      }
    }

    if (!assigned) {
      Species new_species(genome);
      new_species.add_member(e, genome);
      species_.push_back(std::move(new_species));
    }

    size_t idx = registry.index_of(e);
    registry.identity().species_id[idx] =
        assigned ? species_.back().id() : species_.size() - 1;
  }

  for (auto &species : species_) {
    species.refresh_summary();
  }
}

void EvolutionManager::compute_actions(Registry &registry) {
  MOONAI_PROFILE_SCOPE("evolution_compute_actions");
  const auto &living = registry.living_entities();

  std::vector<float> all_inputs;
  all_inputs.reserve(living.size() * SensorSoA::INPUT_COUNT);

  for (Entity e : living) {
    size_t idx = registry.index_of(e);
    const float *input_ptr = registry.sensors().input_ptr(idx);
    all_inputs.insert(all_inputs.end(), input_ptr,
                      input_ptr + SensorSoA::INPUT_COUNT);
  }

  std::vector<float> all_outputs;
  compute_actions_batch(living, all_inputs, all_outputs);

  for (size_t i = 0; i < living.size(); ++i) {
    Vec2 action{all_outputs[i * 2], all_outputs[i * 2 + 1]};
    size_t idx = registry.index_of(living[i]);
    registry.brain().decision_x[idx] = action.x;
    registry.brain().decision_y[idx] = action.y;
  }
}

void EvolutionManager::compute_actions_batch(
    const std::vector<Entity> &entities, const std::vector<float> &all_inputs,
    std::vector<float> &all_outputs) {
  network_cache_.activate_batch(entities, all_inputs, all_outputs,
                                SensorSoA::INPUT_COUNT,
                                SensorSoA::OUTPUT_COUNT);
}

void EvolutionManager::on_entity_destroyed(Entity e) {
  entity_genomes_.erase(e);
  network_cache_.remove(e);
}

Genome *EvolutionManager::genome_for(Entity e) {
  auto it = entity_genomes_.find(e);
  if (it != entity_genomes_.end()) {
    return &it->second;
  }
  return nullptr;
}

const Genome *EvolutionManager::genome_for(Entity e) const {
  auto it = entity_genomes_.find(e);
  if (it != entity_genomes_.end()) {
    return &it->second;
  }
  return nullptr;
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

bool EvolutionManager::launch_gpu_neural(gpu::GpuBatch &gpu_batch,
                                         std::size_t agent_count) {
  MOONAI_PROFILE_SCOPE("gpu_neural", gpu_batch.stream());

  if (!gpu_network_cache_) {
    spdlog::error("GPU neural cache not initialized");
    return false;
  }

  // Get entities from GPU mapping (in GPU buffer order) and filter to only
  // those with networks. Also collect their GPU buffer indices.
  std::vector<std::pair<Entity, int>> network_entities_with_indices;
  network_entities_with_indices.reserve(agent_count);

  for (uint32_t gpu_idx = 0; gpu_idx < agent_count; ++gpu_idx) {
    Entity e = gpu_batch.agent_mapping().entity_at(gpu_idx);
    if (e != INVALID_ENTITY && network_cache_.has(e)) {
      network_entities_with_indices.emplace_back(e, static_cast<int>(gpu_idx));
    }
  }

  if (network_entities_with_indices.empty()) {
    spdlog::warn("No entities with neural networks found in GPU batch");
    return true;
  }

  // Rebuild GPU cache if networks changed
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
    spdlog::debug("Rebuilding GPU network cache for {} network entities",
                  network_entities_with_indices.size());
    gpu_network_cache_->build_from(network_cache_,
                                   network_entities_with_indices);
  }

  // Launch kernel - only for entities with networks
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
