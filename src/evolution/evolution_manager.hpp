#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"
#include "evolution/genome.hpp"

#include <vector>

namespace moonai {

struct AppState;
struct AgentRegistry;
#ifdef MOONAI_ENABLE_CUDA
namespace gpu {
class GpuBatch;
class GpuNetworkCache;
} // namespace gpu
#endif

class EvolutionManager {
public:
  explicit EvolutionManager(const SimulationConfig &config);
  ~EvolutionManager();

  void initialize(AppState &state, int num_inputs, int num_outputs);

  void seed_initial_population(AppState &state);

  bool run_inference(AppState &state);
  void post_step(AppState &state);
  void refresh_species(AppState &state);

  void enable_gpu(AppState &state, bool use_gpu);

private:
  Genome create_initial_genome(AgentRegistry &registry, Random &rng) const;
  Genome create_child_genome(AgentRegistry &registry, Random &rng, const Genome &parent_a,
                             const Genome &parent_b) const;
  uint32_t create_offspring(AppState &state, AgentRegistry &registry, uint32_t parent_a, uint32_t parent_b,
                            Vec2 spawn_position);
  void initialize_population(AgentRegistry &registry) const;
  void compute_actions_for_population(AgentRegistry &registry, const std::vector<float> &sensors,
                                      std::vector<float> &decisions_out) const;
  void refresh_population_species(AgentRegistry &registry) const;
#ifdef MOONAI_ENABLE_CUDA
  bool launch_gpu_neural(AppState &state, gpu::GpuBatch &gpu_batch);
#endif
  void reproduce_population(AppState &state, AgentRegistry &registry);

  const SimulationConfig &config_;
  int num_inputs_ = 0;
  int num_outputs_ = 0;
};

} // namespace moonai
