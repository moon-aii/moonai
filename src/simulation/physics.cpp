#include "simulation/physics.hpp"
#include "core/profiler.hpp"
#include "simulation/environment.hpp"
#include "simulation/predator.hpp"

#include <cmath>
#include <limits>

namespace moonai {

std::vector<float> SensorInput::to_vector() const {
    return {
        nearest_predator_dist, nearest_predator_angle,
        nearest_prey_dist, nearest_prey_angle,
        nearest_food_dist, nearest_food_angle,
        energy_level, speed_x, speed_y,
        local_predator_density, local_prey_density,
        wall_left, wall_right, wall_top, wall_bottom
    };
}

void SensorInput::write_to(float* buffer) const {
    buffer[0]  = nearest_predator_dist;
    buffer[1]  = nearest_predator_angle;
    buffer[2]  = nearest_prey_dist;
    buffer[3]  = nearest_prey_angle;
    buffer[4]  = nearest_food_dist;
    buffer[5]  = nearest_food_angle;
    buffer[6]  = energy_level;
    buffer[7]  = speed_x;
    buffer[8]  = speed_y;
    buffer[9]  = local_predator_density;
    buffer[10] = local_prey_density;
    buffer[11] = wall_left;
    buffer[12] = wall_right;
    buffer[13] = wall_top;
    buffer[14] = wall_bottom;
}

namespace {

float normalize_angle(float dx, float dy) {
    return std::atan2(dy, dx) / 3.14159265f;  // maps to [-1, 1]
}

} // anonymous namespace

SensorInput Physics::build_sensors(
    const Agent& agent,
    const std::vector<std::unique_ptr<Agent>>& agents,
    const std::vector<Food>& food,
    const SpatialGrid& grid,
    const SpatialGrid& food_grid,
    float world_width, float world_height,
    float max_energy,
    bool has_walls) {

    thread_local std::vector<AgentId> nearby_ids;
    thread_local std::vector<AgentId> nearby_food_ids;
    const Vec2 pos = agent.position();
    const float vision = agent.vision_range();
    grid.query_radius_into(pos, vision, nearby_ids);
    food_grid.query_radius_into(pos, vision, nearby_food_ids);

    return build_sensors_from_candidates(
        agent,
        agents,
        food,
        nearby_ids,
        nearby_food_ids,
        world_width,
        world_height,
        max_energy,
        has_walls);
}

SensorInput Physics::build_sensors_from_candidates(
    const Agent& agent,
    const std::vector<std::unique_ptr<Agent>>& agents,
    const std::vector<Food>& food,
    const std::vector<AgentId>& nearby_agent_ids,
    const std::vector<AgentId>& nearby_food_ids,
    float world_width, float world_height,
    float max_energy,
    bool has_walls) {

    SensorInput s;
    float vision = agent.vision_range();
    float vision_sq = vision * vision;
    Vec2 pos = agent.position();

    // Wrap-aware difference: takes the shortest path across world boundaries.
    auto wrap_diff = [&](Vec2 d) -> Vec2 {
        if (!has_walls) {
            if (std::abs(d.x) > world_width  * 0.5f)
                d.x = (d.x > 0.0f) ? d.x - world_width  : d.x + world_width;
            if (std::abs(d.y) > world_height * 0.5f)
                d.y = (d.y > 0.0f) ? d.y - world_height : d.y + world_height;
        }
        return d;
    };

    float nearest_pred_dist_sq = std::numeric_limits<float>::max();
    float nearest_prey_dist_sq = std::numeric_limits<float>::max();
    Vec2 nearest_pred_dir = {0.0f, 0.0f};
    Vec2 nearest_prey_dir = {0.0f, 0.0f};
    int local_predators = 0;
    int local_prey = 0;

    for (AgentId nid : nearby_agent_ids) {
        if (nid >= agents.size()) continue;
        const auto& other = agents[nid];
        if (!other->alive() || other->id() == agent.id()) continue;

        Vec2 diff = wrap_diff(other->position() - pos);
        float dist_sq = diff.x * diff.x + diff.y * diff.y;
        if (dist_sq > vision_sq) continue;

        if (other->type() == AgentType::Predator) {
            ++local_predators;
            if (dist_sq < nearest_pred_dist_sq) {
                nearest_pred_dist_sq = dist_sq;
                nearest_pred_dir = diff;
            }
        } else {
            ++local_prey;
            if (dist_sq < nearest_prey_dist_sq) {
                nearest_prey_dist_sq = dist_sq;
                nearest_prey_dir = diff;
            }
        }
    }

    // Nearest predator
    if (nearest_pred_dist_sq < std::numeric_limits<float>::max()) {
        s.nearest_predator_dist = std::sqrt(nearest_pred_dist_sq) / vision;
        s.nearest_predator_angle = normalize_angle(nearest_pred_dir.x, nearest_pred_dir.y);
    }

    // Nearest prey
    if (nearest_prey_dist_sq < std::numeric_limits<float>::max()) {
        s.nearest_prey_dist = std::sqrt(nearest_prey_dist_sq) / vision;
        s.nearest_prey_angle = normalize_angle(nearest_prey_dir.x, nearest_prey_dir.y);
    }

    // Nearest food (use spatial grid for O(1) per agent instead of O(n))
    float nearest_food_dist_sq = std::numeric_limits<float>::max();
    Vec2 nearest_food_dir = {0.0f, 0.0f};
    for (AgentId fi : nearby_food_ids) {
        if (fi >= food.size()) continue;
        const auto& f = food[fi];
        if (!f.active) continue;
        Vec2 diff = wrap_diff(f.position - pos);
        float dist_sq = diff.x * diff.x + diff.y * diff.y;
        if (dist_sq < nearest_food_dist_sq) {
            nearest_food_dist_sq = dist_sq;
            nearest_food_dir = diff;
        }
    }
    if (nearest_food_dist_sq < std::numeric_limits<float>::max()) {
        s.nearest_food_dist = std::sqrt(nearest_food_dist_sq) / vision;
        s.nearest_food_angle = normalize_angle(nearest_food_dir.x, nearest_food_dir.y);
    }

    // Self state (normalize against 2× initial so energy from eating is visible)
    s.energy_level = std::clamp(agent.energy() / (max_energy * 2.0f), 0.0f, 1.0f);
    float max_speed = agent.speed();
    if (max_speed > 0.0f) {
        s.speed_x = std::clamp(agent.velocity().x / max_speed, -1.0f, 1.0f);
        s.speed_y = std::clamp(agent.velocity().y / max_speed, -1.0f, 1.0f);
    }

    // Density (normalized by approximate max)
    float max_density = 10.0f;
    s.local_predator_density = std::clamp(local_predators / max_density, 0.0f, 1.0f);
    s.local_prey_density = std::clamp(local_prey / max_density, 0.0f, 1.0f);

    // Wall proximity (only meaningful in Clamp/wall boundary mode)
    if (has_walls) {
        s.wall_left   = std::clamp(pos.x / vision, 0.0f, 1.0f);
        s.wall_right  = std::clamp((world_width  - pos.x) / vision, 0.0f, 1.0f);
        s.wall_top    = std::clamp(pos.y / vision, 0.0f, 1.0f);
        s.wall_bottom = std::clamp((world_height - pos.y) / vision, 0.0f, 1.0f);
    }
    // else: wall sensors stay 0.0f (no walls in wrap mode)

    return s;
}

std::vector<Physics::KillEvent> Physics::process_attacks(
    std::vector<std::unique_ptr<Agent>>& agents,
    const SpatialGrid& grid,
    float attack_range) {
    std::vector<std::size_t> predator_indices;
    predator_indices.reserve(agents.size());
    std::vector<std::vector<AgentId>> nearby_agent_ids(agents.size());

    for (size_t i = 0; i < agents.size(); ++i) {
        if (!agents[i]->alive() || agents[i]->type() != AgentType::Predator) {
            continue;
        }
        predator_indices.push_back(i);
        grid.query_radius_into(agents[i]->position(), attack_range, nearby_agent_ids[i]);
    }

    return process_attacks_from_candidates(agents, nearby_agent_ids, predator_indices, attack_range);
}

std::vector<Physics::KillEvent> Physics::process_attacks_from_candidates(
    std::vector<std::unique_ptr<Agent>>& agents,
    const std::vector<std::vector<AgentId>>& nearby_agent_ids,
    const std::vector<std::size_t>& predator_indices,
    float attack_range) {
    std::vector<KillEvent> kills;
    const float attack_range_sq = attack_range * attack_range;

    for (std::size_t predator_index : predator_indices) {
        if (predator_index >= agents.size()) {
            continue;
        }
        auto& agent = agents[predator_index];
        if (!agent->alive() || agent->type() != AgentType::Predator) {
            continue;
        }

        auto* predator = static_cast<Predator*>(agent.get());
        const auto& nearby = nearby_agent_ids[predator_index];
        MOONAI_PROFILE_INC(ProfileCounter::AttackChecks,
                           static_cast<std::int64_t>(nearby.size()));

        for (AgentId nid : nearby) {
            if (nid >= agents.size()) continue;
            auto& target = agents[nid];
            if (!target->alive() || target->type() != AgentType::Prey) continue;

            Vec2 diff = target->position() - predator->position();
            float dist_sq = diff.x * diff.x + diff.y * diff.y;
            if (dist_sq <= attack_range_sq) {
                target->set_alive(false);
                predator->add_kill();
                kills.push_back({predator->id(), target->id()});
                break;  // one kill per tick per predator
            }
        }
    }

    return kills;
}

} // namespace moonai
