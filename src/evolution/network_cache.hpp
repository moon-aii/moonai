#pragma once
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"
#include "simulation/entity.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace moonai {

// Storage for variable-topology neural networks
// Lives outside ECS but references entities by stable handles
class NetworkCache {
public:
  // Create network for entity from genome
  void assign(Entity e, const Genome &genome,
              const std::string &activation_func);

  // Get network for entity (nullptr if not found)
  NeuralNetwork *get(Entity e) const;

  // Remove network (called when entity dies)
  void remove(Entity e);

  // Check if entity has network
  bool has(Entity e) const;

  // Activate network and return outputs
  std::vector<float> activate(Entity e, const std::vector<float> &inputs) const;

  // Batch activation: Activate multiple networks efficiently
  void activate_batch(const std::vector<Entity> &entities,
                      const std::vector<float> &all_inputs,
                      std::vector<float> &all_outputs, int inputs_per_network,
                      int outputs_per_network);

  // GPU batching: build CSR-formatted network data for all living entities
  struct GpuBatchData {
    std::vector<float> node_values;        // Flattened activations
    std::vector<float> connection_weights; // CSR format
    std::vector<int> topology_offsets;     // Per-entity network layout
    std::vector<Entity> entity_to_gpu;     // Mapping: GPU index -> Entity
  };
  GpuBatchData
  prepare_gpu_batch(const std::vector<Entity> &living_entities) const;

  // Invalidate GPU cache (call after mutation/crossover)
  void invalidate_gpu_cache() {
    gpu_cache_dirty_ = true;
  }

  // Cleanup dead entities
  void prune_dead(const std::vector<Entity> &living);

  // Clear all networks
  void clear();

  // Get count of stored networks
  size_t size() const {
    return networks_.size();
  }
  bool empty() const {
    return networks_.empty();
  }

  // Get all entities with networks
  std::vector<Entity> entities() const;

private:
  std::unordered_map<Entity, std::unique_ptr<NeuralNetwork>, EntityHash>
      networks_;

  // GPU cache
  mutable GpuBatchData gpu_cache_;
  mutable bool gpu_cache_dirty_ = true;
};

} // namespace moonai
