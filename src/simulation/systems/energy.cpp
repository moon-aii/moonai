#include "simulation/systems/energy.hpp"
#include "core/profiler_macros.hpp"
#include "simulation/components.hpp"

namespace moonai {

EnergySystem::EnergySystem(float predator_energy_cost, float prey_energy_cost,
                           float max_age, float max_energy)
    : predator_energy_cost_(predator_energy_cost),
      prey_energy_cost_(prey_energy_cost), max_age_(max_age),
      max_energy_(max_energy) {}

void EnergySystem::update(Registry &registry) {
  MOONAI_PROFILE_SCOPE("energy_update");
  auto &vitals = registry.vitals();
  auto &identity = registry.identity();
  auto &stats = registry.stats();

  const size_t count = registry.size();

  for (size_t i = 0; i < count; ++i) {
    if (!vitals.alive[i]) {
      continue;
    }

    vitals.age[i]++;

    float energy_cost = (identity.type[i] == IdentitySoA::TYPE_PREDATOR)
                            ? predator_energy_cost_
                            : prey_energy_cost_;

    vitals.energy[i] -= energy_cost;

    bool died_of_starvation = vitals.energy[i] <= 0.0f;
    bool died_of_age = (max_age_ > 0.0f && vitals.age[i] >= max_age_);
    if (died_of_starvation || died_of_age) {
      vitals.alive[i] = 0;
    }

    if (vitals.reproduction_cooldown[i] > 0) {
      vitals.reproduction_cooldown[i]--;
    }
  }
}

} // namespace moonai
