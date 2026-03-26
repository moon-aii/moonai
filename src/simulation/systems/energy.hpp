#pragma once
#include "simulation/system.hpp"

namespace moonai {

class EnergySystem : public System {
public:
  EnergySystem(float predator_energy_cost, float prey_energy_cost,
               float max_age, float max_energy);

  void update(Registry &registry, float dt) override;
  const char *name() const override {
    return "EnergySystem";
  }

private:
  float predator_energy_cost_;
  float prey_energy_cost_;
  float max_age_;
  float max_energy_;
};

} // namespace moonai