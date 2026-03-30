#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

#include <vector>

namespace moonai::simulation_detail {

void build_sensors(AgentSoA &self_agents, const PositionSoA &self_positions,
                   const AgentSoA &predator_agents,
                   const PositionSoA &predator_positions,
                   const AgentSoA &prey_agents,
                   const PositionSoA &prey_positions,
                   const FoodStore &food_store, const SimulationConfig &config);
void update_vitals(AgentSoA &agents, const SimulationConfig &config);
void process_food(PreyRegistry &prey_registry, FoodStore &food_store,
                  const SimulationConfig &config,
                  std::vector<int> &food_consumed_by);
void process_combat(PredatorRegistry &predator_registry,
                    PreyRegistry &prey_registry, const SimulationConfig &config,
                    std::vector<int> &killed_by,
                    std::vector<uint32_t> &kill_counts);
void apply_movement(PositionSoA &positions, AgentSoA &agents,
                    const SimulationConfig &config);
void collect_food_events(PreyRegistry &prey_registry,
                         const FoodStore &food_store,
                         const std::vector<uint8_t> &was_food_active,
                         const std::vector<int> &food_consumed_by,
                         std::vector<SimEvent> &events);
void collect_combat_events(PredatorRegistry &predator_registry,
                           const PreyRegistry &prey_registry,
                           const std::vector<int> &killed_by,
                           const std::vector<uint32_t> &kill_counts,
                           std::vector<SimEvent> &events);
void collect_death_events(const PredatorRegistry &predator_registry,
                          const std::vector<uint8_t> &was_alive,
                          std::vector<SimEvent> &events);
void collect_death_events(const PreyRegistry &prey_registry,
                          const std::vector<uint8_t> &was_alive,
                          std::vector<SimEvent> &events);
void accumulate_events(EventCounters &counters,
                       const std::vector<SimEvent> &events);

} // namespace moonai::simulation_detail
