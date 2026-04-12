#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "evolution/genome.hpp"
#include "visualization/overlay.hpp"
#include "visualization/renderer.hpp"

#include <unordered_map>
#include <vector>

namespace moonai {

struct FrameSnapshot {
  int world_width = 0;
  int world_height = 0;
  std::vector<RenderFood> foods;
  std::vector<RenderAgent> predators;
  std::vector<RenderAgent> prey;
  bool has_selected_vision = false;
  uint32_t selected_agent_id = 0;
  Vec2 selected_position;
  float selected_vision_range = 0.0f;
  std::vector<RenderLine> sensor_lines;
  OverlayStats overlay_stats;
  const Genome *selected_genome = nullptr;
  std::unordered_map<std::uint32_t, float> selected_node_activations;
};

FrameSnapshot build_frame_snapshot(const AppState &state, const AppConfig &config);

} // namespace moonai
