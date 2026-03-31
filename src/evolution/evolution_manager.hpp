#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"
#include "evolution/genome.hpp"

#include <memory>
#include <vector>

namespace moonai {

struct AppState;
struct PopulationEvolutionState;
namespace gpu {
class GpuBatch;
class GpuNetworkCache;
} // namespace gpu

class EvolutionManager {
public:
  explicit EvolutionManager(const SimulationConfig &config);
  ~EvolutionManager();

  void initialize(AppState &state, int num_inputs, int num_outputs);

  void seed_initial_population(AppState &state);

  uint32_t create_predator_offspring(AppState &state, uint32_t parent_a,
                                     uint32_t parent_b, Vec2 spawn_position);
  uint32_t create_prey_offspring(AppState &state, uint32_t parent_a,
                                 uint32_t parent_b, Vec2 spawn_position);

  void refresh_species(AppState &state);

  void compute_actions(AppState &state,
                       const std::vector<float> &predator_sensors,
                       const std::vector<float> &prey_sensors,
                       std::vector<float> &predator_decisions,
                       std::vector<float> &prey_decisions);

  void on_predator_destroyed(AppState &state, uint32_t e);
  void on_predator_moved(AppState &state, uint32_t from, uint32_t to);
  void on_prey_destroyed(AppState &state, uint32_t e);
  void on_prey_moved(AppState &state, uint32_t from, uint32_t to);

  void enable_gpu(bool use_gpu);
  bool gpu_enabled() const {
    return use_gpu_;
  }

  // GPU neural inference (called by SimulationManager during GPU step)
  bool launch_gpu_neural(AppState &state, gpu::GpuBatch &gpu_batch);

private:
  Genome create_initial_genome(PopulationEvolutionState &population,
                               Random &rng) const;
  Genome create_child_genome(PopulationEvolutionState &population, Random &rng,
                             const Genome &parent_a,
                             const Genome &parent_b) const;
  void initialize_population(PopulationEvolutionState &population) const;
  void compute_actions_for_population(PopulationEvolutionState &population,
                                      AgentRegistry &agents,
                                      const std::vector<float> &sensors,
                                      std::vector<float> &decisions_out) const;
  void refresh_population_species(PopulationEvolutionState &population,
                                  AgentRegistry &agents) const;
  void on_population_destroyed(PopulationEvolutionState &population,
                               uint32_t entity);
  void on_population_moved(PopulationEvolutionState &population, uint32_t from,
                           uint32_t to);

  const SimulationConfig &config_;
  int num_inputs_ = 0;
  int num_outputs_ = 0;
  bool use_gpu_ = false;

  std::unique_ptr<gpu::GpuNetworkCache> predator_gpu_network_cache_;
  std::unique_ptr<gpu::GpuNetworkCache> prey_gpu_network_cache_;
};

} // namespace moonai
