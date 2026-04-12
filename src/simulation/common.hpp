#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

namespace moonai {

class EvolutionManager;

namespace common {

void run(AppState &state, EvolutionManager &evolution, const SimulationConfig &config);

} // namespace common

} // namespace moonai