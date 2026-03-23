#include "simulation/simulation_manager.hpp"
#include "core/profiler.hpp"
#include "simulation/predator.hpp"
#include "simulation/prey.hpp"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>

namespace moonai {

SimulationManager::SimulationManager(const SimulationConfig& config)
    : config_(config)
    , rng_(config.seed != 0
           ? config.seed
           : static_cast<std::uint64_t>(
                 std::chrono::steady_clock::now().time_since_epoch().count()))
    , environment_(config)
    , grid_(config.grid_width, config.grid_height,
            std::max(config.vision_range * 0.5f, 1.0f))
    , food_grid_(config.grid_width, config.grid_height,
                 std::max(config.vision_range * 0.5f, 1.0f)) {
}

void SimulationManager::initialize() {
    agents_.clear();
    current_tick_ = 0;

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
    count_alive();

    spdlog::info("Simulation initialized: {} predators, {} prey, {} food (seed: {})",
                 config_.predator_count, config_.prey_count, config_.food_count, rng_.seed());
}

void SimulationManager::tick(float dt) {
    ScopedTimer timer(ProfileEvent::SimulationTick);
    last_events_.clear();
    rebuild_spatial_grid();
    rebuild_food_grid();

    // Update agents (age increment)
    {
        ScopedTimer section(ProfileEvent::AgentUpdate);
        for (auto& agent : agents_) {
            if (agent->alive()) {
                agent->update(dt);
            }
        }
    }

    // Process interactions
    process_energy(dt);
    process_food();
    process_attacks();

    // Respawn food
    environment_.tick_food(rng_, config_.food_respawn_rate);

    // Apply boundary conditions
    {
        ScopedTimer section(ProfileEvent::BoundaryApply);
        for (auto& agent : agents_) {
            if (agent->alive()) {
                agent->set_position(environment_.apply_boundary(agent->position()));
            }
        }
    }

    // Check for energy death
    {
        ScopedTimer section(ProfileEvent::DeathCheck);
        for (auto& agent : agents_) {
            if (agent->alive() && agent->is_dead()) {
                agent->set_alive(false);
            }
        }
    }

    count_alive();
    ++current_tick_;
}

void SimulationManager::reset() {
    initialize();
}

SensorInput SimulationManager::get_sensors(size_t agent_index) const {
    if (agent_index >= agents_.size() || !agents_[agent_index]->alive()) {
        return SensorInput{};
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
    for (size_t i = 0; i < agent_count; ++i) {
        get_sensors(i).write_to(dst + i * SensorInput::SIZE);
    }
}

void SimulationManager::apply_action(size_t agent_index, Vec2 direction, float dt) {
    if (agent_index >= agents_.size() || !agents_[agent_index]->alive()) return;
    agents_[agent_index]->apply_movement(direction, dt);
}

void SimulationManager::rebuild_spatial_grid() {
    ScopedTimer timer(ProfileEvent::RebuildSpatialGrid);
    grid_.clear();
    for (const auto& agent : agents_) {
        if (agent->alive()) {
            grid_.insert(agent->id(), agent->position());
        }
    }
}

void SimulationManager::rebuild_food_grid() {
    ScopedTimer timer(ProfileEvent::RebuildFoodGrid);
    food_grid_.clear();
    const auto& food = environment_.food();
    for (size_t i = 0; i < food.size(); ++i) {
        if (food[i].active) {
            food_grid_.insert(static_cast<AgentId>(i), food[i].position);
        }
    }
}

void SimulationManager::process_energy(float dt) {
    ScopedTimer timer(ProfileEvent::ProcessEnergy);
    for (auto& agent : agents_) {
        if (!agent->alive()) continue;
        // All agents drain energy per tick (cost of living)
        agent->drain_energy(config_.energy_drain_per_tick * dt * static_cast<float>(config_.target_fps));
    }
}

void SimulationManager::process_food() {
    ScopedTimer timer(ProfileEvent::ProcessFood);
    float eat_range = config_.food_pickup_range;
    for (auto& agent : agents_) {
        if (!agent->alive() || agent->type() != AgentType::Prey) continue;
        Profiler::instance().increment(ProfileCounter::FoodEatAttempts);
        if (environment_.try_eat_food(agent->position(), eat_range)) {
            agent->add_energy(config_.energy_gain_from_food);
            agent->add_food();
            Profiler::instance().increment(ProfileCounter::FoodEaten);
            last_events_.push_back({SimEvent::Food, agent->id(), 0, agent->position()});
        }
    }
}

void SimulationManager::process_attacks() {
    ScopedTimer timer(ProfileEvent::ProcessAttacks);
    // Record kills before attack processing
    std::vector<int> kills_before;
    kills_before.reserve(agents_.size());
    for (const auto& agent : agents_) {
        kills_before.push_back(agent->kills());
    }

    // Track which prey were alive before attacks to identify kill events
    std::vector<bool> alive_before;
    alive_before.reserve(agents_.size());
    for (const auto& agent : agents_) {
        alive_before.push_back(agent->alive());
    }

    Physics::process_attacks(agents_, grid_, config_.attack_range);

    // Reward energy for new kills this tick and record kill events
    for (size_t i = 0; i < agents_.size(); ++i) {
        if (agents_[i]->type() != AgentType::Predator) continue;
        int new_kills = agents_[i]->kills() - kills_before[i];
        if (new_kills > 0) {
            Profiler::instance().increment(ProfileCounter::Kills, new_kills);
            if (agents_[i]->alive()) {
                agents_[i]->add_energy(config_.energy_gain_from_kill * static_cast<float>(new_kills));
            }
            // Find which prey this predator killed (newly dead prey nearby)
            for (size_t j = 0; j < agents_.size(); ++j) {
                if (agents_[j]->type() != AgentType::Prey) continue;
                if (alive_before[j] && !agents_[j]->alive()) {
                    last_events_.push_back({SimEvent::Kill, agents_[i]->id(),
                                            agents_[j]->id(), agents_[j]->position()});
                }
            }
        }
    }
}

void SimulationManager::count_alive() {
    ScopedTimer timer(ProfileEvent::CountAlive);
    alive_predators_ = 0;
    alive_prey_ = 0;
    for (const auto& agent : agents_) {
        if (!agent->alive()) continue;
        if (agent->type() == AgentType::Predator) ++alive_predators_;
        else ++alive_prey_;
    }
}

} // namespace moonai
