#include "simulation/simulation_manager.hpp"
#include "core/profiler_types.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <spdlog/spdlog.h>
#include <unordered_set>

#ifdef MOONAI_OPENMP_ENABLED
#include <omp.h>
#endif

namespace moonai {

SimulationManager::SimulationManager(const SimulationConfig &config)
    : config_(config),
      rng_(config.seed != 0
               ? config.seed
               : static_cast<std::uint64_t>(std::chrono::steady_clock::now()
                                                .time_since_epoch()
                                                .count())),
      environment_(config), grid_(config.grid_size, config.grid_size,
                                  std::max(config.vision_range, 1.0f)),
      food_grid_(config.grid_size, config.grid_size,
                 std::max(config.vision_range, 1.0f)) {}

void SimulationManager::initialize() {
  initialize(true);
}

void SimulationManager::initialize(bool log_initialization) {
  agents_.clear();
  agent_slots_.clear();
  next_agent_id_ = 0;
  current_step_ = 0;
  invalidate_neighbor_cache();

  environment_.initialize_food(rng_, config_.food_count);

  rebuild_spatial_grid();
  rebuild_food_grid();
  rebuild_alive_indices();
  rebuild_neighbor_cache(config_.vision_range);
  count_alive();

  if (log_initialization) {
    spdlog::info("Simulation initialized: {} food pellets (seed: {})",
                 config_.food_count, rng_.seed());
  }
}

AgentId SimulationManager::spawn_agent(std::unique_ptr<Agent> agent) {
  const AgentId id = agent->id();
  if (id != next_agent_id_) {
    next_agent_id_ = std::max(next_agent_id_, static_cast<AgentId>(id + 1));
  } else {
    ++next_agent_id_;
  }
  const std::size_t slot = agents_.size();
  grid_.insert(id, agent->position());
  agent_slots_[id] = slot;
  agents_.push_back(std::move(agent));
  rebuild_alive_indices();
  count_alive();
  neighbor_cache_.valid = false;
  return id;
}

std::size_t SimulationManager::slot_for_id(AgentId id) const {
  const auto it = agent_slots_.find(id);
  return it == agent_slots_.end() ? agents_.size() : it->second;
}

Agent *SimulationManager::agent_by_id(AgentId id) {
  const std::size_t slot = slot_for_id(id);
  return slot < agents_.size() ? agents_[slot].get() : nullptr;
}

const Agent *SimulationManager::agent_by_id(AgentId id) const {
  const std::size_t slot = slot_for_id(id);
  return slot < agents_.size() ? agents_[slot].get() : nullptr;
}

void SimulationManager::step(float dt) {
  MOONAI_PROFILE_SCOPE(ProfileEvent::SimulationStep);
  last_events_.clear();
  rebuild_alive_indices();
  rebuild_neighbor_cache(config_.vision_range);

  // Update agents (age increment)
  {
    MOONAI_PROFILE_SCOPE(ProfileEvent::AgentUpdate);
#pragma omp parallel for schedule(static) if (MOONAI_OPENMP_ENABLED)
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
  environment_.step_food_deterministic(
      config_.seed, current_step_, config_.food_respawn_rate, respawned_food);
  for (AgentId food_id : respawned_food) {
    const auto &food = environment_.food();
    if (food_id < food.size() && food[food_id].active) {
      food_grid_.insert(food_id, food[food_id].position);
    }
  }

  // Apply boundary conditions
  {
    MOONAI_PROFILE_SCOPE(ProfileEvent::BoundaryApply);
#pragma omp parallel for schedule(static) if (MOONAI_OPENMP_ENABLED)
    for (size_t idx = 0; idx < alive_indices_.size(); ++idx) {
      auto &agent = agents_[alive_indices_[idx]];
      const Vec2 bounded = environment_.apply_boundary(agent->position());
      agent->set_position(bounded);
      grid_.update(agent->id(), bounded);
    }
  }

  process_step_deaths();

  rebuild_alive_indices();
  rebuild_neighbor_cache(config_.vision_range);
  count_alive();
  ++current_step_;
}

void SimulationManager::reset() {
  initialize(false);
}

SensorInput SimulationManager::get_sensors(size_t agent_index) const {
  if (agent_index >= agents_.size() || !agents_[agent_index]->alive()) {
    return SensorInput{};
  }

  if (neighbor_cache_enabled_ && neighbor_cache_.valid &&
      agent_index < neighbor_cache_.nearby_agents.size() &&
      agent_index < neighbor_cache_.nearby_food.size()) {
    return Physics::build_sensors_from_candidates(
        *agents_[agent_index], agents_, environment_.food(),
        neighbor_cache_.nearby_agents[agent_index], agent_slots_,
        neighbor_cache_.nearby_food[agent_index],
        static_cast<float>(config_.grid_size),
        static_cast<float>(config_.grid_size), config_.initial_energy,
        config_.boundary_mode == BoundaryMode::Clamp);
  }

  return Physics::build_sensors(
      *agents_[agent_index], agents_, environment_.food(), grid_, food_grid_,
      agent_slots_, static_cast<float>(config_.grid_size),
      static_cast<float>(config_.grid_size), config_.initial_energy,
      config_.boundary_mode == BoundaryMode::Clamp);
}

void SimulationManager::write_sensors_flat(float *dst,
                                           size_t agent_count) const {
#pragma omp parallel for schedule(dynamic) if (MOONAI_OPENMP_ENABLED)
  for (size_t i = 0; i < agent_count; ++i) {
    get_sensors(i).write_to(dst + i * SensorInput::SIZE);
  }
}

void SimulationManager::apply_action(size_t agent_index, Vec2 direction,
                                     float dt) {
  if (agent_index >= agents_.size() || !agents_[agent_index]->alive())
    return;
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

#pragma omp parallel for schedule(dynamic) if (MOONAI_OPENMP_ENABLED)
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
  for (const auto &agent : agents_) {
    if (agent->alive()) {
      grid_.insert(agent->id(), agent->position());
    }
  }
}

void SimulationManager::rebuild_food_grid() {
  MOONAI_PROFILE_SCOPE(ProfileEvent::RebuildFoodGrid);
  food_grid_.clear();
  const auto &food = environment_.food();
  for (size_t i = 0; i < food.size(); ++i) {
    if (food[i].active) {
      food_grid_.insert(static_cast<AgentId>(i), food[i].position);
    }
  }
}

void SimulationManager::process_energy(float dt) {
  MOONAI_PROFILE_SCOPE(ProfileEvent::ProcessEnergy);
#pragma omp parallel for schedule(static) if (MOONAI_OPENMP_ENABLED)
  for (size_t idx = 0; idx < alive_indices_.size(); ++idx) {
    auto &agent = agents_[alive_indices_[idx]];
    // All agents drain energy per step (cost of living)
    agent->drain_energy(config_.energy_drain_per_step * dt *
                        static_cast<float>(config_.target_fps));
  }
}

void SimulationManager::process_food() {
  MOONAI_PROFILE_SCOPE(ProfileEvent::ProcessFood);
  float eat_range = config_.food_pickup_range;
  for (size_t prey_index : alive_prey_indices_) {
    auto &agent = agents_[prey_index];
    if (!agent->alive())
      continue;
    AgentId eaten_food = 0;
    bool ate_food = false;
    if (neighbor_cache_enabled_ && neighbor_cache_.valid &&
        prey_index < neighbor_cache_.nearby_food.size()) {
      ate_food = environment_.try_eat_food(
          agent->position(), eat_range, neighbor_cache_.nearby_food[prey_index],
          &eaten_food);
    } else {
      thread_local std::vector<AgentId> nearby_food;
      food_grid_.query_radius_into(agent->position(), eat_range, nearby_food);
      ate_food = environment_.try_eat_food(agent->position(), eat_range,
                                           nearby_food, &eaten_food);
    }
    if (ate_food) {
      food_grid_.remove(eaten_food);
      agent->add_energy(config_.energy_gain_from_food);
      agent->add_food();
      last_events_.push_back(SimEvent{SimEvent::Food, agent->id(), eaten_food,
                                      0, 0, agent->position()});
    }
  }
}

void SimulationManager::process_attacks() {
  MOONAI_PROFILE_SCOPE(ProfileEvent::ProcessAttacks);
  auto kills = (neighbor_cache_enabled_ && neighbor_cache_.valid)
                   ? Physics::process_attacks_from_candidates(
                         agents_, neighbor_cache_.nearby_agents, agent_slots_,
                         alive_predator_indices_, config_.attack_range)
                   : Physics::process_attacks(agents_, grid_, agent_slots_,
                                              config_.attack_range);

  for (const auto &kill : kills) {
    // Reward energy to the killer
    if (Agent *killer = agent_by_id(kill.killer);
        killer != nullptr && killer->alive()) {
      killer->add_energy(config_.energy_gain_from_kill);
    }
    // Record the kill event
    const auto victim_slot = slot_for_id(kill.victim);
    if (victim_slot < agents_.size()) {
      last_events_.push_back(SimEvent{SimEvent::Kill, kill.killer, kill.victim,
                                      0, 0, agents_[victim_slot]->position()});
    }
  }
}

void SimulationManager::process_step_deaths() {
  MOONAI_PROFILE_SCOPE(ProfileEvent::DeathCheck);
  std::vector<std::size_t> dead_slots;
  for (std::size_t idx : alive_indices_) {
    auto &agent = agents_[idx];
    if (!agent->alive() || !agent->is_dead()) {
      continue;
    }
    const Vec2 death_pos = agent->position();
    const AgentId death_id = agent->id();
    agent->set_alive(false);
    grid_.remove(death_id);
    last_events_.push_back(
        SimEvent{SimEvent::Death, death_id, death_id, 0, 0, death_pos});
    dead_slots.push_back(idx);
  }
  std::sort(dead_slots.begin(), dead_slots.end());
  for (auto it = dead_slots.rbegin(); it != dead_slots.rend(); ++it) {
    remove_agent_slot(*it);
  }
}

std::vector<SimulationManager::ReproductionPair>
SimulationManager::find_reproduction_pairs() const {
  std::vector<ReproductionPair> pairs;
  std::unordered_set<AgentId> used;

  for (std::size_t idx : alive_indices_) {
    const auto &agent = agents_[idx];
    if (!agent->alive() ||
        agent->energy() < config_.reproduction_energy_threshold ||
        agent->age() < config_.min_reproductive_age_steps ||
        agent->reproduction_cooldown() > 0 ||
        used.find(agent->id()) != used.end()) {
      continue;
    }

    std::vector<AgentId> nearby;
    grid_.query_radius_into(agent->position(), config_.mate_range, nearby);

    AgentId best_mate = std::numeric_limits<AgentId>::max();
    float best_dist_sq = config_.mate_range * config_.mate_range;
    for (AgentId mate_id : nearby) {
      if (mate_id == agent->id() || used.find(mate_id) != used.end()) {
        continue;
      }
      const Agent *mate = agent_by_id(mate_id);
      if (mate == nullptr) {
        continue;
      }
      if (!mate->alive() || mate->type() != agent->type() ||
          mate->energy() < config_.reproduction_energy_threshold ||
          mate->age() < config_.min_reproductive_age_steps ||
          mate->reproduction_cooldown() > 0) {
        continue;
      }

      const Vec2 diff = mate->position() - agent->position();
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_mate = mate_id;
      }
    }

    if (best_mate == std::numeric_limits<AgentId>::max()) {
      continue;
    }

    const Vec2 a = agent->position();
    const Agent *mate = agent_by_id(best_mate);
    if (mate == nullptr) {
      continue;
    }
    const Vec2 b = mate->position();
    pairs.push_back(
        {agent->id(), best_mate, {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f}});
    used.insert(agent->id());
    used.insert(best_mate);
  }

  return pairs;
}

void SimulationManager::count_alive() {
  MOONAI_PROFILE_SCOPE(ProfileEvent::CountAlive);
  alive_predators_ = static_cast<int>(alive_predator_indices_.size());
  alive_prey_ = static_cast<int>(alive_prey_indices_.size());
}

void SimulationManager::remove_agent_slot(std::size_t slot) {
  if (slot >= agents_.size()) {
    return;
  }

  const AgentId removed_id = agents_[slot]->id();
  agent_slots_.erase(removed_id);

  const std::size_t last = agents_.size() - 1;
  if (slot != last) {
    std::swap(agents_[slot], agents_[last]);
    agent_slots_[agents_[slot]->id()] = slot;
  }
  agents_.pop_back();
  invalidate_neighbor_cache();
}

} // namespace moonai
