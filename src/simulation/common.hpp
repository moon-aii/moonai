#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"

namespace moonai {

namespace simulation {
namespace common {

void post_step(AppState &state, const SimulationConfig &config);

} // namespace common
} // namespace simulation

} // namespace moonai
