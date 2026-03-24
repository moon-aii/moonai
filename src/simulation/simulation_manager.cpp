#include "simulation/simulation_manager.hpp"
#include "core/profiler.hpp"
#include "simulation/predator.hpp"
#include "simulation/prey.hpp"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>

#ifdef MOONAI_OPENMP_ENABLED
#include <omp.h>
#endif

namespace moonai {

SimulationManager::SimulationManager(const SimulationConfig& config)
    : config_(config)
    , rng_(config.seed != 0
           ? config.seed
           : static_cast<std::uint64_t>(
                 std::chrono::steady_clock::now().time_since_epoch().count()))
    , environment_(config)
    , grid_(config.grid_width, config.grid_height,
            std::max(config.vision_range, 1.0f))
    , food_grid_(config.grid_width, config.grid_height,
                 std::max(config.vision_range, 1.0f)) {
}

void SimulationManager::initialize() {
    initialize(true);
}

void SimulationManager::initialize(bool log_initialization) {
    agents_.clear();
    current_tick_ = 0;
    invalidate_neighbor_cache();

    AgentId next_id = 0;

    for (int i = 0; i < config_.predator_count; ++i) {
        Vec2 pos{rng_.next_float(0, static_cast<float>(config_.grid_width)),
                 rng_.next_float(0, static_cast<float>(config_.grid_height))};
        agents_.push_back(std::make_unique<Predator>(
            next_id++, pos, config_.predator_speed,
            config_.vision_range, config_.initial_energy,
            config_.attack_range));
    }

    for (int i = 0; i < config_.prey_count; ++i) {
        Vec2 pos{rng_.next_float(0, static_cast<float>(config_.grid_width)),
                 rng_.next_float(0, static_cast<float>(config_.grid_height))};
        agents_.push_back(std::make_unique<Prey>(
            next_id++, pos, config_.prey_speed,
            config_.vision_range, config_.initial_energy));
    }

    environment_.initialize_food(rng_, config_.food_count);

    rebuild_spatial_grid();
    rebuild_food_grid();
    rebuild_alive_indices();
    rebuild_neighbor_cache(config_.vision_range);
    count_alive();

    if (log_initialization) {
        spdlog::info("Simulation initialized: {} predators, {} prey, {} food (seed: {})",
                     config_.predator_count, config_.prey_count, config_.food_count, rng_.seed());
    }
}

void SimulationManager::tick(float dt) {
    MOONAI_PROFILE_SCOPE(ProfileEvent::SimulationTick);
    last_events_.clear();
    rebuild_alive_indices();
    rebuild_neighbor_cache(config_.vision_range);

    // Update agents (age increment)
    {
        MOONAI_PROFILE_SCOPE(ProfileEvent::AgentUpdate);
        #pragma omp parallel for schedule(static) if(MOONAI_OPENMP_ENABLED)
        for (size_t idx = 0; idx < alive_indices_.size(); ++idx) {
            agents_[alive_indices_[idx]]->update(dt);
        }
    }

    // Process interactions
    process_energy(dt);
    process_food();
    process_attacks();

    // Respawn food
    std::vector<AgentId> respawned_food;
    environment_.tick_food_deterministic(config_.seed, current_tick_,
                                         config_.food_respawn_rate, respawned_food);
    for (AgentId food_id : respawned_food) {
        const auto& food = environment_.food();
        if (food_id < food.size() && food[food_id].active) {
            food_grid_.insert(food_id, food[food_id].position);
        }
    }

    // Apply boundary conditions
    {
        MOONAI_PROFILE_SCOPE(ProfileEvent::BoundaryApply);
        #pragma omp parallel for schedule(static) if(MOONAI_OPENMP_ENABLED)
        for (size_t idx = 0; idx < alive_indices_.size(); ++idx) {
            auto& agent = agents_[alive_indices_[idx]];
            const Vec2 bounded = environment_.apply_boundary(agent->position());
            agent->set_position(bounded);
            grid_.update(agent->id(), bounded);
        }
    }

    // Check for energy death
    {
        MOONAI_PROFILE_SCOPE(ProfileEvent::DeathCheck);
        #pragma omp parallel for schedule(static) if(MOONAI_OPENMP_ENABLED)
        for (size_t idx = 0; idx < alive_indices_.size(); ++idx) {
            auto& agent = agents_[alive_indices_[idx]];
            if (agent->is_dead()) {
                agent->set_alive(false);
                grid_.remove(agent->id());
            }
        }
    }

    rebuild_alive_indices();
    rebuild_neighbor_cache(config_.vision_range);
    count_alive();
    ++current_tick_;
}

void SimulationManager::reset() {
    initialize(false);
}

SensorInput SimulationManager::get_sensors(size_t agent_index) const {
    if (agent_index >= agents_.size() || !agents_[agent_index]->alive()) {
        return SensorInput{};
    }

    if (neighbor_cache_enabled_ && neighbor_cache_.valid
        && agent_index < neighbor_cache_.nearby_agents.size()
        && agent_index < neighbor_cache_.nearby_food.size()) {
        return Physics::build_sensors_from_candidates(
            *agents_[agent_index],
            agents_,
            environment_.food(),
            neighbor_cache_.nearby_agents[agent_index],
            neighbor_cache_.nearby_food[agent_index],
            static_cast<float>(config_.grid_width),
            static_cast<float>(config_.grid_height),
            config_.initial_energy,
            config_.boundary_mode == BoundaryMode::Clamp);
    }

    return Physics::build_sensors(
        *agents_[agent_index],
        agents_,
        environment_.food(),
        grid_,
        food_grid_,
        static_cast<float>(config_.grid_width),
        static_cast<float>(config_.grid_height),
        config_.initial_energy,
        config_.boundary_mode == BoundaryMode::Clamp);
}

void SimulationManager::write_sensors_flat(float* dst, size_t agent_count) const {
    #pragma omp parallel for schedule(dynamic) if(MOONAI_OPENMP_ENABLED)
    for (size_t i = 0; i < agent_count; ++i) {
        get_sensors(i).write_to(dst + i * SensorInput::SIZE);
    }
}

void SimulationManager::apply_action(size_t agent_index, Vec2 direction, float dt) {
    if (agent_index >= agents_.size() || !agents_[agent_index]->alive()) return;
    agents_[agent_index]->apply_movement(direction, dt);
    grid_.update(agents_[agent_index]->id(), agents_[agent_index]->position());
    neighbor_cache_.valid = false;
}

void SimulationManager::refresh_state() {
    rebuild_spatial_grid();
    rebuild_food_grid();
    rebuild_alive_indices();
    rebuild_neighbor_cache(config_.vision_range);
    count_alive();
}

void SimulationManager::rebuild_alive_indices() {
    alive_indices_.clear();
    alive_predator_indices_.clear();
    alive_prey_indices_.clear();

    alive_indices_.reserve(agents_.size());
    alive_predator_indices_.reserve(static_cast<size_t>(config_.predator_count));
    alive_prey_indices_.reserve(static_cast<size_t>(config_.prey_count));

    for (size_t i = 0; i < agents_.size(); ++i) {
        if (!agents_[i]->alive()) {
            continue;
        }
        alive_indices_.push_back(i);
        if (agents_[i]->type() == AgentType::Predator) {
            alive_predator_indices_.push_back(i);
        } else {
            alive_prey_indices_.push_back(i);
        }
    }
}

void SimulationManager::invalidate_neighbor_cache() {
    neighbor_cache_.valid = false;
}

void SimulationManager::rebuild_neighbor_cache(float radius) {
    if (!neighbor_cache_enabled_) {
        neighbor_cache_.nearby_agents.clear();
        neighbor_cache_.nearby_food.clear();
        neighbor_cache_.valid = false;
        return;
    }
    neighbor_cache_.nearby_agents.clear();
    neighbor_cache_.nearby_agents.resize(agents_.size());
    neighbor_cache_.nearby_food.clear();
    neighbor_cache_.nearby_food.resize(agents_.size());

    #pragma omp parallel for schedule(dynamic) if(MOONAI_OPENMP_ENABLED)
    for (size_t idx = 0; idx < alive_indices_.size(); ++idx) {
        const size_t i = alive_indices_[idx];
        const Vec2 pos = agents_[i]->position();
        grid_.query_radius_into(pos, radius, neighbor_cache_.nearby_agents[i]);
        food_grid_.query_radius_into(pos, radius, neighbor_cache_.nearby_food[i]);
    }

    neighbor_cache_.valid = true;
}

void SimulationManager::rebuild_spatial_grid() {
    MOONAI_PROFILE_SCOPE(ProfileEvent::RebuildSpatialGrid);
    grid_.clear();
    for (const auto& agent : agents_) {
        if (agent->alive()) {
            grid_.insert(agent->id(), agent->position());
        }
    }
}

void SimulationManager::rebuild_food_grid() {
    MOONAI_PROFILE_SCOPE(ProfileEvent::RebuildFoodGrid);
    food_grid_.clear();
    const auto& food = environment_.food();
    for (size_t i = 0; i < food.size(); ++i) {
        if (food[i].active) {
            food_grid_.insert(static_cast<AgentId>(i), food[i].position);
        }
    }
}

void SimulationManager::process_energy(float dt) {
    MOONAI_PROFILE_SCOPE(ProfileEvent::ProcessEnergy);
    #pragma omp parallel for schedule(static) if(MOONAI_OPENMP_ENABLED)
    for (size_t idx = 0; idx < alive_indices_.size(); ++idx) {
        auto& agent = agents_[alive_indices_[idx]];
        // All agents drain energy per tick (cost of living)
        agent->drain_energy(config_.energy_drain_per_tick * dt * static_cast<float>(config_.target_fps));
    }
}

void SimulationManager::process_food() {
    MOONAI_PROFILE_SCOPE(ProfileEvent::ProcessFood);
    float eat_range = config_.food_pickup_range;
    for (size_t prey_index : alive_prey_indices_) {
        auto& agent = agents_[prey_index];
        if (!agent->alive()) continue;
        MOONAI_PROFILE_INC(ProfileCounter::FoodEatAttempts);
        AgentId eaten_food = 0;
        bool ate_food = false;
        if (neighbor_cache_enabled_ && neighbor_cache_.valid
            && prey_index < neighbor_cache_.nearby_food.size()) {
            ate_food = environment_.try_eat_food(agent->position(), eat_range,
                                                 neighbor_cache_.nearby_food[prey_index], &eaten_food);
        } else {
            thread_local std::vector<AgentId> nearby_food;
            food_grid_.query_radius_into(agent->position(), eat_range, nearby_food);
            ate_food = environment_.try_eat_food(agent->position(), eat_range, nearby_food, &eaten_food);
        }
        if (ate_food) {
            food_grid_.remove(eaten_food);
            agent->add_energy(config_.energy_gain_from_food);
            agent->add_food();
            MOONAI_PROFILE_INC(ProfileCounter::FoodEaten);
            last_events_.push_back({SimEvent::Food, agent->id(), eaten_food, agent->position()});
        }
    }
}

void SimulationManager::process_attacks() {
    MOONAI_PROFILE_SCOPE(ProfileEvent::ProcessAttacks);
    auto kills = (neighbor_cache_enabled_ && neighbor_cache_.valid)
        ? Physics::process_attacks_from_candidates(
            agents_,
            neighbor_cache_.nearby_agents,
            alive_predator_indices_,
            config_.attack_range)
        : Physics::process_attacks(agents_, grid_, config_.attack_range);

    MOONAI_PROFILE_INC(ProfileCounter::Kills, static_cast<std::int64_t>(kills.size()));

    for (const auto& kill : kills) {
        // Reward energy to the killer
        if (kill.killer < agents_.size() && agents_[kill.killer]->alive()) {
            agents_[kill.killer]->add_energy(config_.energy_gain_from_kill);
        }
        // Record the kill event
        if (kill.victim < agents_.size()) {
            last_events_.push_back({SimEvent::Kill, kill.killer,
                                    kill.victim, agents_[kill.victim]->position()});
        }
    }
}

void SimulationManager::count_alive() {
    MOONAI_PROFILE_SCOPE(ProfileEvent::CountAlive);
    alive_predators_ = static_cast<int>(alive_predator_indices_.size());
    alive_prey_ = static_cast<int>(alive_prey_indices_.size());
}

} // namespace moonai
