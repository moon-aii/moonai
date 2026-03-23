#pragma once

#include "simulation/agent.hpp"
#include "simulation/spatial_grid.hpp"

#include <vector>
#include <memory>

namespace moonai {

// Sensor input vector layout for the neural network.
// All values normalized to roughly [0, 1] or [-1, 1].
struct SensorInput {
    // Nearest predator (relative)
    float nearest_predator_dist = -1.0f;  // -1=none in range, 0=touching, 1=at vision limit
    float nearest_predator_angle = 0.0f;  // [-1, 1] mapped from [-pi, pi]

    // Nearest prey (relative)
    float nearest_prey_dist = -1.0f;      // -1=none in range
    float nearest_prey_angle = 0.0f;

    // Nearest food (relative, for prey)
    float nearest_food_dist = -1.0f;      // -1=none in range
    float nearest_food_angle = 0.0f;

    // Self state
    float energy_level = 1.0f;  // [0, 1]
    float speed_x = 0.0f;      // [-1, 1]
    float speed_y = 0.0f;      // [-1, 1]

    // Density around agent
    float local_predator_density = 0.0f;  // [0, 1]
    float local_prey_density = 0.0f;      // [0, 1]

    // Wall proximity (for clamp mode)
    float wall_left = 1.0f;
    float wall_right = 1.0f;
    float wall_top = 1.0f;
    float wall_bottom = 1.0f;

    static constexpr int SIZE = 15;
    std::vector<float> to_vector() const;
    void write_to(float* buffer) const;
};

class Environment;
struct Food;

class Physics {
public:
    // Build sensor inputs for one agent
    static SensorInput build_sensors(
        const Agent& agent,
        const std::vector<std::unique_ptr<Agent>>& agents,
        const std::vector<Food>& food,
        const SpatialGrid& grid,
        const SpatialGrid& food_grid,
        float world_width, float world_height,
        float max_energy,
        bool has_walls);

    // Process predator attacks: returns list of killed prey IDs
    static std::vector<AgentId> process_attacks(
        std::vector<std::unique_ptr<Agent>>& agents,
        const SpatialGrid& grid,
        float attack_range);
};

} // namespace moonai
