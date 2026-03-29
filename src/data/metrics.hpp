#pragma once

namespace moonai {

struct AppState;

namespace metrics {

void refresh_live(AppState &state);
void record_report(AppState &state);

} // namespace metrics

} // namespace moonai
