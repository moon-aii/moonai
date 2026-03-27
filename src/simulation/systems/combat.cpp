#include "simulation/systems/combat.hpp"
#include "core/profiler_macros.hpp"
#include "simulation/components.hpp"
#include <cmath>
#include <limits>

namespace moonai {

CombatSystem::CombatSystem(const SpatialGridECS &agent_grid, float attack_range)
    : agent_grid_(agent_grid), attack_range_(attack_range),
      attack_range_sq_(attack_range * attack_range) {}

void CombatSystem::update(Registry &registry) {
  MOONAI_PROFILE_SCOPE("combat_update");
  kill_events_.clear();
  food_events_.clear();

  (void)registry;
  process_predator_attacks(registry);
}

void CombatSystem::process_predator_attacks(Registry &registry) {
  auto &positions = registry.positions();
  auto &vitals = registry.vitals();
  auto &identity = registry.identity();
  auto &stats = registry.stats();

  const size_t count = registry.size();

  for (size_t predator_idx = 0; predator_idx < count; ++predator_idx) {
    if (!vitals.alive[predator_idx]) {
      continue;
    }

    if (identity.type[predator_idx] != IdentitySoA::TYPE_PREDATOR) {
      continue;
    }

    Vec2 predator_pos{positions.x[predator_idx], positions.y[predator_idx]};

    auto nearby = agent_grid_.query_radius(predator_pos, attack_range_);

    Entity predator_entity;
    bool found_predator = false;

    const auto &living = registry.living_entities();
    for (size_t i = 0; i < living.size(); ++i) {
      if (registry.index_of(living[i]) == predator_idx) {
        predator_entity = living[i];
        found_predator = true;
        break;
      }
    }

    if (!found_predator) {
      continue;
    }

    for (Entity prey_entity : nearby) {
      size_t prey_idx = registry.index_of(prey_entity);

      if (prey_idx == std::numeric_limits<size_t>::max()) {
        continue;
      }

      if (!vitals.alive[prey_idx]) {
        continue;
      }

      if (identity.type[prey_idx] != IdentitySoA::TYPE_PREY) {
        continue;
      }

      Vec2 prey_pos{positions.x[prey_idx], positions.y[prey_idx]};
      float dx = prey_pos.x - predator_pos.x;
      float dy = prey_pos.y - predator_pos.y;
      float dist_sq = dx * dx + dy * dy;

      if (dist_sq <= attack_range_sq_) {
        vitals.alive[prey_idx] = 0;
        stats.kills[predator_idx]++;

        kill_events_.push_back({predator_entity, prey_entity});

        break;
      }
    }
  }
}

} // namespace moonai
