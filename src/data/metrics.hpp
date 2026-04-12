#pragma once

namespace moonai {

struct AppState;

namespace metrics {

void begin_step(AppState &state);
void refresh_live(AppState &state);
void finalize_step(AppState &state);
void record_report(AppState &state);

} // namespace metrics

} // namespace moonai
