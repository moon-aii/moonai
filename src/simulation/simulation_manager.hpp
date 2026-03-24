#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "simulation/environment.hpp"
#include "simulation/agent.hpp"
#include "simulation/spatial_grid.hpp"
#include "simulation/physics.hpp"

#include <vector>
#include <memory>
#include <functional>
#include <cstddef>

namespace moonai {

// Discrete interaction event recorded each tick
struct SimEvent {
    enum Type : uint8_t { Kill, Food };
    Type type;
    AgentId agent_id;    // predator (kill) or prey (food)
    AgentId target_id;   // prey (kill) or food index (food)
    Vec2 position;       // where the event occurred
};

class SimulationManager {
public:
    explicit SimulationManager(const SimulationConfig& config);

    void initialize();
    void tick(float dt);
    void reset();

    int current_tick() const { return current_tick_; }
    std::vector<std::unique_ptr<Agent>>& agents() { return agents_; }
    const std::vector<std::unique_ptr<Agent>>& agents() const { return agents_; }
    Environment& environment() { return environment_; }
    const Environment& environment() const { return environment_; }
    const SpatialGrid& spatial_grid() const { return grid_; }
    const SpatialGrid& food_grid() const { return food_grid_; }

    int alive_predators() const { return alive_predators_; }
    int alive_prey() const { return alive_prey_; }
    const std::vector<std::size_t>& alive_agent_indices() const { return alive_indices_; }
    const std::vector<std::size_t>& alive_predator_indices() const { return alive_predator_indices_; }
    const std::vector<std::size_t>& alive_prey_indices() const { return alive_prey_indices_; }

    // Interaction events that occurred during the last tick() call
    const std::vector<SimEvent>& last_events() const { return last_events_; }

    // Get sensor inputs for all agents (indexed by position in agents_ vector)
    SensorInput get_sensors(size_t agent_index) const;
    void write_sensors_flat(float* dst, size_t agent_count) const;

    // Apply neural network output to an agent
    void apply_action(size_t agent_index, Vec2 direction, float dt);
    void set_neighbor_cache_enabled(bool enabled) { neighbor_cache_enabled_ = enabled; }
    void refresh_state();

private:
    void initialize(bool log_initialization);

    struct QueryCache {
        std::vector<std::vector<AgentId>> nearby_agents;
        std::vector<std::vector<AgentId>> nearby_food;
        bool valid = false;
    };

    void rebuild_alive_indices();
    void invalidate_neighbor_cache();
    void rebuild_neighbor_cache(float radius);
    void rebuild_spatial_grid();
    void rebuild_food_grid();
    void process_energy(float dt);
    void process_food();
    void process_attacks();
    void count_alive();

    SimulationConfig config_;
    Random rng_;
    Environment environment_;
    SpatialGrid grid_;
    SpatialGrid food_grid_;
    std::vector<std::unique_ptr<Agent>> agents_;
    std::vector<SimEvent> last_events_;
    std::vector<std::size_t> alive_indices_;
    std::vector<std::size_t> alive_predator_indices_;
    std::vector<std::size_t> alive_prey_indices_;
    QueryCache neighbor_cache_;
    bool neighbor_cache_enabled_ = true;
    int current_tick_ = 0;
    int alive_predators_ = 0;
    int alive_prey_ = 0;
};

} // namespace moonai
