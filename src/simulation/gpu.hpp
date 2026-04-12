#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
namespace moonai {

namespace simulation {
namespace gpu {

bool prepare_step(AppState &state, const SimulationConfig &config);
bool resolve_step(AppState &state, const SimulationConfig &config);

} // namespace gpu
} // namespace simulation

} // namespace moonai
