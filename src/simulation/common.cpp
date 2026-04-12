#include "simulation/common.hpp"

namespace moonai::simulation::common {

void post_step(AppState &state, const SimulationConfig &config) {
  state.predator.compact();
  state.prey.compact();

  state.food.respawn_step(config, state.runtime.step, state.runtime.rng.seed());
}

} // namespace moonai::simulation::common
