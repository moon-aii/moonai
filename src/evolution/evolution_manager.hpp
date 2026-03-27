#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"
#include "evolution/genome.hpp"
#include "evolution/mutation.hpp"
#include "evolution/network_cache.hpp"
#include "evolution/neural_network.hpp"
#include "evolution/species.hpp"
#include "simulation/entity.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonai {

class Registry;
namespace gpu {
class GpuBatchECS;
class GpuNetworkCache;
} // namespace gpu

class EvolutionManager {
public:
  explicit EvolutionManager(const SimulationConfig &config, Random &rng);
  ~EvolutionManager();

  void initialize(int num_inputs, int num_outputs);

  Genome create_initial_genome() const;
  Genome create_child_genome(const Genome &parent_a,
                             const Genome &parent_b) const;

  void seed_initial_population_ecs(Registry &registry);

  Entity create_offspring_ecs(Registry &registry, Entity parent_a,
                              Entity parent_b, Vec2 spawn_position);

  void refresh_fitness_ecs(const Registry &registry);
  void refresh_species_ecs(Registry &registry);

  void compute_actions_ecs(Registry &registry);

  void on_entity_destroyed(Entity e);

  NetworkCache &network_cache() {
    return network_cache_;
  }
  const NetworkCache &network_cache() const {
    return network_cache_;
  }

  Genome *genome_for(Entity e);
  const Genome *genome_for(Entity e) const;

  const std::vector<Species> &species() const {
    return species_;
  }
  int species_count() const {
    return static_cast<int>(species_.size());
  }

  void get_fitness_by_type_ecs(const Registry &registry, float &best_predator,
                               float &avg_predator, float &best_prey,
                               float &avg_prey) const;

  void update_config(const SimulationConfig &cfg) {
    config_ = cfg;
  }
  void enable_gpu(bool use_gpu);
  bool gpu_enabled() const {
    return use_gpu_;
  }

  // GPU neural inference (called by SimulationManager during GPU step)
  void launch_gpu_neural(gpu::GpuBatchECS &gpu_batch, std::size_t agent_count);

private:
  SimulationConfig config_;
  Random &rng_;
  InnovationTracker tracker_;
  std::vector<Species> species_;
  int num_inputs_ = 0;
  int num_outputs_ = 0;
  bool use_gpu_ = false;
  int species_refresh_step_ = -1;

  std::unordered_map<Entity, Genome, EntityHash> entity_genomes_;

  // Entity -> NeuralNetwork mapping (variable topology, separate cache)
  NetworkCache network_cache_;

  // GPU network cache for CSR-formatted batched inference
  std::unique_ptr<gpu::GpuNetworkCache> gpu_network_cache_;
};

} // namespace moonai
