#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

namespace moonai {

namespace simulation {
namespace cpu {

bool prepare_step(AppState &state, const SimulationConfig &config);
bool resolve_step(AppState &state, const SimulationConfig &config);

} // namespace cpu
} // namespace simulation

} // namespace moonai
