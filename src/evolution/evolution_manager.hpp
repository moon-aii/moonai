#pragma once

#include "core/config.hpp"
#include "core/types.hpp"
#include "evolution/genome.hpp"

#include <memory>
#include <vector>

namespace moonai {

struct AppState;
namespace gpu {
class GpuBatch;
class GpuNetworkCache;
} // namespace gpu

class EvolutionManager {
public:
  explicit EvolutionManager(const SimulationConfig &config);
  ~EvolutionManager();

  void initialize(AppState &state, int num_inputs, int num_outputs);

  Genome create_initial_genome(AppState &state) const;
  Genome create_child_genome(AppState &state, const Genome &parent_a,
                             const Genome &parent_b) const;

  void seed_initial_population(AppState &state);

  uint32_t create_offspring(AppState &state, uint32_t parent_a, uint32_t parent_b,
                          Vec2 spawn_position);

  void refresh_species(AppState &state);

  void compute_actions(AppState &state);
  void compute_actions_batch(std::size_t entity_count, AppState &state,
                             const std::vector<float> &all_inputs,
                             std::vector<float> &all_outputs);

  void on_entity_destroyed(AppState &state, uint32_t e);
  void on_entity_moved(AppState &state, uint32_t from, uint32_t to);

  Genome *genome_for(AppState &state, uint32_t e);
  const Genome *genome_for(const AppState &state, uint32_t e) const;

  int species_count(const AppState &state) const;

  void enable_gpu(bool use_gpu);
  bool gpu_enabled() const {
    return use_gpu_;
  }

  // GPU neural inference (called by SimulationManager during GPU step)
  bool launch_gpu_neural(AppState &state, gpu::GpuBatch &gpu_batch,
                         std::size_t agent_count);

private:
  const SimulationConfig &config_;
  int num_inputs_ = 0;
  int num_outputs_ = 0;
  bool use_gpu_ = false;

  // GPU network cache for CSR-formatted batched inference
  std::unique_ptr<gpu::GpuNetworkCache> gpu_network_cache_;
};

} // namespace moonai
