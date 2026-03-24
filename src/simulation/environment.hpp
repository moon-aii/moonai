#pragma once

#include "core/types.hpp"
#include "core/config.hpp"
#include "core/random.hpp"

#include <cstdint>
#include <vector>

namespace moonai {

struct Food {
    Vec2 position;
    bool active = true;
};

class Environment {
public:
    explicit Environment(const SimulationConfig& config);

    int width() const { return width_; }
    int height() const { return height_; }
    BoundaryMode boundary_mode() const { return boundary_mode_; }

    Vec2 apply_boundary(Vec2 pos) const;

    // Food management
    void initialize_food(Random& rng, int count);
    void tick_food(Random& rng, float respawn_rate);
    void tick_food(Random& rng, float respawn_rate, std::vector<AgentId>& respawned_ids);
    void tick_food_deterministic(std::uint64_t seed, int tick_index, float respawn_rate,
                                 std::vector<AgentId>& respawned_ids);
    bool try_eat_food(Vec2 position, float range);
    bool try_eat_food(Vec2 position, float range, const std::vector<AgentId>& candidate_ids,
                      AgentId* eaten_id = nullptr);
    const std::vector<Food>& food() const { return food_; }
    std::vector<Food>& mutable_food() { return food_; }

private:
    int width_;
    int height_;
    BoundaryMode boundary_mode_;
    std::vector<Food> food_;
    int max_food_;
};

} // namespace moonai
