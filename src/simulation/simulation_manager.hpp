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
    const std::vector<std::unique_ptr<Agent>>& agents() const { return agents_; }
    const Environment& environment() const { return environment_; }

    int alive_predators() const { return alive_predators_; }
    int alive_prey() const { return alive_prey_; }

    // Interaction events that occurred during the last tick() call
    const std::vector<SimEvent>& last_events() const { return last_events_; }

    // Get sensor inputs for all agents (indexed by position in agents_ vector)
    SensorInput get_sensors(size_t agent_index) const;
    void write_sensors_flat(float* dst, size_t agent_count) const;

    // Apply neural network output to an agent
    void apply_action(size_t agent_index, Vec2 direction, float dt);

private:
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
    int current_tick_ = 0;
    int alive_predators_ = 0;
    int alive_prey_ = 0;
};

} // namespace moonai
