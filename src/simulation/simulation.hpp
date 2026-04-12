#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

namespace moonai {

namespace simulation {

void initialize(AppState &state, const SimulationConfig &config);
bool prepare_step(AppState &state, const SimulationConfig &config);
bool resolve_step(AppState &state, const SimulationConfig &config);
void post_step(AppState &state, const SimulationConfig &config);

} // namespace simulation

} // namespace moonai
