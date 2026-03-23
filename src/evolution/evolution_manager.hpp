#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"
#include "evolution/species.hpp"
#include "evolution/mutation.hpp"
#include "simulation/simulation_manager.hpp"

#include <vector>
#include <memory>
#include <functional>

#ifdef MOONAI_ENABLE_CUDA
namespace moonai::gpu { class GpuBatch; }
#endif

namespace moonai {

class LuaRuntime;  // forward declaration

class EvolutionManager {
public:
    explicit EvolutionManager(const SimulationConfig& config, Random& rng);
    ~EvolutionManager();

    void initialize(int num_inputs, int num_outputs);

    // Run one full generation: evaluate all agents, compute fitness, evolve
    void evaluate_generation(SimulationManager& sim);

    // Evolve population based on current fitness values
    void evolve();

    const std::vector<Genome>& population() const { return population_; }
    std::vector<Genome>& population() { return population_; }
    int generation() const { return generation_; }
    const std::vector<Species>& species() const { return species_; }
    int species_count() const { return static_cast<int>(species_.size()); }

    // Build neural networks from current population for simulation use
    const std::vector<std::unique_ptr<NeuralNetwork>>& networks() const { return networks_; }

    // Compute and assign fitness for each genome from simulation results
    void compute_fitness(const SimulationManager& sim);

    // Per-tick callback (called after each sim.tick() in the CPU/headless path)
    using TickCallback = std::function<void(int tick, const SimulationManager&)>;
    void set_tick_callback(TickCallback cb) { tick_callback_ = std::move(cb); }
    void clear_tick_callback() { tick_callback_ = nullptr; }

    // Lua runtime (for scripted fitness / hooks)
    void set_lua_runtime(LuaRuntime* rt) { lua_runtime_ = rt; }

    // Update config (e.g. from Lua hook overrides)
    void update_config(const SimulationConfig& cfg) { config_ = cfg; }

    // GPU acceleration
    void enable_gpu(bool use_gpu);
    bool gpu_enabled() const { return use_gpu_; }

    // Checkpoint serialization
    void save_checkpoint(const std::string& path, const Random& rng) const;
    bool load_checkpoint(const std::string& path, Random& rng);

    // Returns pointer to genome at population index, or nullptr if out of range
    const Genome* genome_at(int idx) const;

    // Assign species IDs to agents based on current speciation
    void assign_species_ids(SimulationManager& sim) const;

private:
    void build_networks();
    void speciate();
    void reproduce();
    void remove_stagnant_species();

    SimulationConfig config_;
    Random& rng_;
    InnovationTracker tracker_;
    std::vector<Genome> population_;
    std::vector<Species> species_;
    std::vector<std::unique_ptr<NeuralNetwork>> networks_;
    int generation_ = 0;
    int num_inputs_ = 0;
    int num_outputs_ = 0;

    LuaRuntime* lua_runtime_ = nullptr;

    bool use_gpu_ = false;
#ifdef MOONAI_ENABLE_CUDA
    std::unique_ptr<moonai::gpu::GpuBatch> gpu_batch_;
#endif

    TickCallback tick_callback_;
};

} // namespace moonai
