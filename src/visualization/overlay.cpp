#include "visualization/overlay.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <queue>
#include <spdlog/spdlog.h>
#include <tuple>
#include <unordered_map>

namespace moonai {

bool UIOverlay::initialize() {
  constexpr const char *kBundledFontPath = "assets/fonts/JetBrainsMono-Regular.ttf";

  if (!font_.openFromFile(kBundledFontPath)) {
    spdlog::warn("Failed to load bundled font '{}'. UI overlay will be disabled.", kBundledFontPath);
    font_loaded_ = false;
    return false;
  }

  font_loaded_ = true;
  spdlog::info("UI font loaded from bundled asset: {}", kBundledFontPath);
  return true;
}

void UIOverlay::draw(sf::RenderTarget &target, const OverlayStats &stats, const Genome *selected_genome) {
  if (!font_loaded_)
    return;

  // Get the current view to draw UI in screen space
  sf::View ui_view = target.getDefaultView();
  sf::View current_view = target.getView();
  target.setView(ui_view);

  // Draw left column (FPS/stats panel only)
  draw_left_column(target, stats);

  // Draw right column (simulation stats widgets)
  draw_right_column(target, stats);

  // Selected agent panel (bottom-left)
  float panel_width = 220.0f;
  float margin = 25.0f;
  char buf[128];
  float line_h = 18.0f;

  if (stats.selected_agent >= 0) {
    float sel_y = target.getDefaultView().getSize().y - margin - 130.0f;
    draw_panel(target, margin, sel_y, panel_width, 120.0f);

    float sx = margin + 8.0f;
    float sy = sel_y + 6.0f;

    std::snprintf(buf, sizeof(buf), "Agent #%d", stats.selected_agent);
    draw_text(target, buf, sx, sy, 14, sf::Color(ui::TITLE_R, ui::TITLE_G, ui::TITLE_B));
    sy += line_h;

    std::snprintf(buf, sizeof(buf), "Energy: %.1f  Age: %d", stats.selected_energy, stats.selected_age);
    draw_text(target, buf, sx, sy, 13);
    sy += line_h;

    std::snprintf(buf, sizeof(buf), "Kills: %d  Food: %d", stats.selected_kills, stats.selected_food_eaten);
    draw_text(target, buf, sx, sy, 13);
    sy += line_h;

    std::snprintf(buf, sizeof(buf), "Complexity: %d", stats.selected_genome_complexity);
    draw_text(target, buf, sx, sy, 13, sf::Color(ui::MUTED_R, ui::MUTED_G, ui::MUTED_B));
  }

  // NN topology panel (bottom-right, anchored to right edge)
  if (selected_genome) {
    draw_nn_panel(target, *selected_genome);
  }

  // Restore the camera view
  target.setView(current_view);
}

void UIOverlay::draw_panel(sf::RenderTarget &target, float x, float y, float w, float h) {
  panel_bg_.setSize({w, h});
  panel_bg_.setPosition({x, y});
  panel_bg_.setFillColor(sf::Color(visual::PANEL_BG_R, visual::PANEL_BG_G, visual::PANEL_BG_B, visual::PANEL_ALPHA));
  panel_bg_.setOutlineColor(
      sf::Color(visual::PANEL_OUTLINE_R, visual::PANEL_OUTLINE_G, visual::PANEL_OUTLINE_B, visual::PANEL_ALPHA));
  panel_bg_.setOutlineThickness(1.0f);
  target.draw(panel_bg_);
}

void UIOverlay::draw_text(sf::RenderTarget &target, const std::string &str, float x, float y, unsigned int size,
                          sf::Color color) {
  sf::Text text(font_, str, size);
  text.setPosition({x, y});
  text.setFillColor(color);
  target.draw(text);
}

void UIOverlay::push_population(int predators, int prey, int food) {
  population_history_.push_back(std::make_tuple(predators, prey, food));
  // No limit - unlimited growth as requested
}

void UIOverlay::draw_left_column(sf::RenderTarget &target, const OverlayStats &stats) {
  constexpr float PANEL_WIDTH = 300.0f;
  constexpr float MARGIN = 25.0f;

  float x = MARGIN;
  float y = MARGIN;

  // First widget: Basic info (step, FPS, speed) - stays on left
  draw_stats_panel(target, stats, x, y, PANEL_WIDTH);
}

void UIOverlay::draw_right_column(sf::RenderTarget &target, const OverlayStats &stats) {
  constexpr float PANEL_WIDTH = 300.0f;
  constexpr float MARGIN = 25.0f;

  // Position at right side of screen
  float x = target.getDefaultView().getSize().x - PANEL_WIDTH - MARGIN;
  float y = MARGIN;

  // Stats widget: Population counts, species, and events
  draw_stats_widget(target, stats, x, y, PANEL_WIDTH, 170.0f);
  y += 170.0f + MARGIN;

  // Population chart
  draw_population_chart(target, x, y, PANEL_WIDTH, 180.0f);
  y += 180.0f + MARGIN;

  // Energy distribution
  draw_energy_distribution(target, stats, x, y, PANEL_WIDTH, 55.0f);
}

void UIOverlay::draw_stats_panel(sf::RenderTarget &target, const OverlayStats &stats, float x, float y, float w) {
  constexpr float PANEL_H = 90.0f;
  float line_h = 18.0f;

  draw_panel(target, x, y, w, PANEL_H);

  float tx = x + 8.0f;
  float ty = y + 6.0f;
  char buf[128];

  if (!stats.experiment_name.empty()) {
    draw_text(target, stats.experiment_name, tx, ty, 16, sf::Color(ui::TITLE_R, ui::TITLE_G, ui::TITLE_B));
  } else {
    draw_text(target, "MoonAI", tx, ty, 16, sf::Color(ui::TITLE_R, ui::TITLE_G, ui::TITLE_B));
  }
  ty += line_h + 4;

  std::snprintf(buf, sizeof(buf), "Step: %d", stats.step);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "FPS: %.0f", stats.fps);
  draw_text(target, buf, tx, ty, 13, sf::Color(ui::MUTED_R, ui::MUTED_G, ui::MUTED_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Speed: %dx%s", stats.speed_multiplier, stats.paused ? " [PAUSED]" : "");
  draw_text(target, buf, tx, ty, 13,
            stats.paused ? sf::Color(ui::PAUSE_R, ui::PAUSE_G, ui::PAUSE_B)
                         : sf::Color(ui::MUTED_R, ui::MUTED_G, ui::MUTED_B));
}

void UIOverlay::draw_stats_widget(sf::RenderTarget &target, const OverlayStats &stats, float x, float y, float w,
                                  float h) {
  if (!font_loaded_)
    return;

  draw_panel(target, x, y, w, h);
  draw_text(target, "Stats", x + 4.0f, y + 2.0f, 11, sf::Color(ui::TITLE_R, ui::TITLE_G, ui::TITLE_B));

  float tx = x + 8.0f;
  float ty = y + 22.0f;
  float line_h = 18.0f;
  char buf[32];

  // Single column: Population counts and events
  std::snprintf(buf, sizeof(buf), "Predators: %d", stats.alive_predator);
  draw_text(target, buf, tx, ty, 13,
            sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Prey: %d", stats.alive_prey);
  draw_text(target, buf, tx, ty, 13, sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Food: %d", stats.active_food);
  draw_text(target, buf, tx, ty, 13, sf::Color(chart_colors::FOOD_R, chart_colors::FOOD_G, chart_colors::FOOD_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Pred Species: %d", stats.predator_species);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Prey Species: %d", stats.prey_species);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Kills: %d", stats.total_kills);
  draw_text(target, buf, tx, ty, 13, sf::Color(ui::EVENT_KILL_R, ui::EVENT_KILL_G, ui::EVENT_KILL_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Eaten: %d", stats.total_food_eaten);
  draw_text(target, buf, tx, ty, 13, sf::Color(ui::EVENT_FOOD_R, ui::EVENT_FOOD_G, ui::EVENT_FOOD_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Births: %d", stats.total_births);
  draw_text(target, buf, tx, ty, 13, sf::Color(ui::EVENT_BIRTH_R, ui::EVENT_BIRTH_G, ui::EVENT_BIRTH_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Deaths: %d", stats.total_deaths);
  draw_text(target, buf, tx, ty, 13, sf::Color(ui::EVENT_DEATH_R, ui::EVENT_DEATH_G, ui::EVENT_DEATH_B));
}

void UIOverlay::draw_population_chart(sf::RenderTarget &target, float x, float y, float w, float h) {
  if (!font_loaded_ || population_history_.size() < 2)
    return;

  draw_panel(target, x, y, w, h);
  draw_text(target, "Population", x + 4.0f, y + 2.0f, 11, sf::Color(180, 180, 200));

  // Chart area inside panel
  float inner_x = x + 4.0f;
  float inner_y = y + 20.0f;
  float inner_w = w - 8.0f;
  float inner_h = h - 28.0f;

  // Find max population for scaling
  int max_pop = 10;
  for (const auto &t : population_history_) {
    max_pop = std::max(max_pop, std::max({std::get<0>(t), std::get<1>(t), std::get<2>(t)}));
  }

  // Show ALL points - unlimited history, compressed X-axis
  size_t total_points = population_history_.size();

  // Map all points across the entire chart width
  auto map_point = [&](size_t idx, int val) -> sf::Vector2f {
    float px = inner_x + (static_cast<float>(idx) / (total_points - 1)) * inner_w;
    float py = inner_y + inner_h * (1.0f - static_cast<float>(val) / max_pop);
    return {px, py};
  };

  // Draw predator line (orange) - ALL points
  sf::VertexArray pred_line(sf::PrimitiveType::LineStrip, total_points);
  for (size_t i = 0; i < total_points; ++i) {
    pred_line[static_cast<int>(i)].position = map_point(i, std::get<0>(population_history_[i]));
    pred_line[static_cast<int>(i)].color =
        sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B);
  }
  target.draw(pred_line);

  // Draw prey line (cyan) - ALL points
  sf::VertexArray prey_line(sf::PrimitiveType::LineStrip, total_points);
  for (size_t i = 0; i < total_points; ++i) {
    prey_line[static_cast<int>(i)].position = map_point(i, std::get<1>(population_history_[i]));
    prey_line[static_cast<int>(i)].color = sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B);
  }
  target.draw(prey_line);

  // Draw food line (yellow) - ALL points
  sf::VertexArray food_line(sf::PrimitiveType::LineStrip, total_points);
  for (size_t i = 0; i < total_points; ++i) {
    food_line[static_cast<int>(i)].position = map_point(i, std::get<2>(population_history_[i]));
    food_line[static_cast<int>(i)].color = sf::Color(chart_colors::FOOD_R, chart_colors::FOOD_G, chart_colors::FOOD_B);
  }
  target.draw(food_line);

  // Legend
  draw_text(target, "Pred", x + w - 110.0f, y + 4.0f, 10,
            sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B));
  draw_text(target, "Prey", x + w - 70.0f, y + 4.0f, 10,
            sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B));
  draw_text(target, "Food", x + w - 30.0f, y + 4.0f, 10,
            sf::Color(chart_colors::FOOD_R, chart_colors::FOOD_G, chart_colors::FOOD_B));
}

void UIOverlay::draw_energy_distribution(sf::RenderTarget &target, const OverlayStats &stats, float x, float y, float w,
                                         float h) {
  if (!font_loaded_)
    return;

  draw_panel(target, x, y, w, h);
  draw_text(target, "Energy Distribution", x + 4.0f, y + 2.0f, 11, sf::Color(ui::TITLE_R, ui::TITLE_G, ui::TITLE_B));

  float bar_y = y + 22.0f;
  float bar_h = 10.0f;
  float label_width = 20.0f;
  float bar_w = w - 16.0f - label_width;
  float tx = x + 8.0f + label_width;

  // Draw 5 buckets as stacked bars
  // Each bucket is 20% energy range: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
  sf::Color bucket_colors[5] = {
      sf::Color(ui::ENERGY_BUCKET_0_R, ui::ENERGY_BUCKET_0_G,
                ui::ENERGY_BUCKET_0_B), // Dark gray (0-20%)
      sf::Color(ui::ENERGY_BUCKET_1_R, ui::ENERGY_BUCKET_1_G,
                ui::ENERGY_BUCKET_1_B), // Gray (20-40%)
      sf::Color(ui::ENERGY_BUCKET_2_R, ui::ENERGY_BUCKET_2_G,
                ui::ENERGY_BUCKET_2_B), // Light gray (40-60%)
      sf::Color(ui::ENERGY_BUCKET_3_R, ui::ENERGY_BUCKET_3_G,
                ui::ENERGY_BUCKET_3_B), // Lighter gray (60-80%)
      sf::Color(ui::ENERGY_BUCKET_4_R, ui::ENERGY_BUCKET_4_G,
                ui::ENERGY_BUCKET_4_B) // White-ish (80-100%)
  };

  // Predator energy bar
  float cx = tx;
  for (int i = 0; i < 5; ++i) {
    float seg_w = stats.predator_energy_dist[i] * bar_w;
    if (seg_w > 0.5f) {
      sf::RectangleShape seg({seg_w, bar_h});
      seg.setPosition({cx, bar_y});
      seg.setFillColor(bucket_colors[i]);
      target.draw(seg);
    }
    cx += seg_w;
  }

  // Labels inside panel, left of bars (at predator bar position)
  float label_x = x + 10.0f;
  draw_text(target, "P", label_x, bar_y, 10,
            sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B));

  // Prey energy bar (below)
  bar_y += bar_h + 4.0f;
  cx = tx;
  for (int i = 0; i < 5; ++i) {
    float seg_w = stats.prey_energy_dist[i] * bar_w;
    if (seg_w > 0.5f) {
      sf::RectangleShape seg({seg_w, bar_h});
      seg.setPosition({cx, bar_y});
      seg.setFillColor(bucket_colors[i]);
      target.draw(seg);
    }
    cx += seg_w;
  }

  // Label for prey bar
  draw_text(target, "Y", label_x, bar_y, 10,
            sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B));
}

void UIOverlay::set_activations(const std::unordered_map<std::uint32_t, float> &vals) {
  node_activations_ = vals;
}

void UIOverlay::draw_nn_panel(sf::RenderTarget &target, const Genome &genome) {
  constexpr float PANEL_W = 250.0f;
  constexpr float PANEL_H = 300.0f;
  constexpr float MARGIN = 10.0f;
  constexpr float NODE_R = 5.0f;

  sf::Vector2f view_size = target.getDefaultView().getSize();

  float cx = view_size.x - PANEL_W - MARGIN;
  float cy = view_size.y - PANEL_H - MARGIN;

  draw_panel(target, cx, cy, PANEL_W, PANEL_H);
  draw_text(target, "Network", cx + 4.0f, cy + 2.0f, 11, sf::Color(ui::TITLE_R, ui::TITLE_G, ui::TITLE_B));

  const auto &nodes = genome.nodes();
  const auto &conns = genome.connections();

  // Assign layer depth via BFS from input/bias nodes
  std::unordered_map<std::uint32_t, int> depth;
  int max_depth = 0;
  std::queue<std::uint32_t> bfs_queue;

  for (const auto &n : nodes) {
    if (n.type == NodeType::Input || n.type == NodeType::Bias) {
      depth[n.id] = 0;
      bfs_queue.push(n.id);
    } else if (n.type == NodeType::Output) {
      depth[n.id] = 2; // outputs start at depth 2; will be pushed by hidden if deeper
    }
  }
  while (!bfs_queue.empty()) {
    auto nid = bfs_queue.front();
    bfs_queue.pop();
    for (const auto &c : conns) {
      if (!c.enabled || c.in_node != nid)
        continue;
      int d = depth[nid] + 1;
      auto it = depth.find(c.out_node);
      if (it == depth.end() || it->second < d) {
        depth[c.out_node] = d;
        max_depth = std::max(max_depth, d);
        bfs_queue.push(c.out_node);
      }
    }
  }
  // Force output nodes to max_depth + 1
  for (const auto &n : nodes) {
    if (n.type == NodeType::Output)
      depth[n.id] = max_depth + 1;
  }
  int num_layers = max_depth + 2;

  // Group nodes by layer
  std::unordered_map<int, std::vector<std::uint32_t>> layers;
  for (const auto &n : nodes) {
    layers[depth[n.id]].push_back(n.id);
  }

  // Compute pixel position per node
  float inner_x = cx + 10.0f;
  float inner_y = cy + 20.0f;
  float inner_w = PANEL_W - 20.0f;
  float inner_h = PANEL_H - 28.0f;

  std::unordered_map<std::uint32_t, sf::Vector2f> pos;
  for (auto &[layer, layer_nodes] : layers) {
    float lx = inner_x + (num_layers <= 1 ? inner_w / 2.0f : static_cast<float>(layer) / (num_layers - 1) * inner_w);
    int cnt = static_cast<int>(layer_nodes.size());
    for (int k = 0; k < cnt; ++k) {
      float ly = inner_y + (cnt <= 1 ? inner_h / 2.0f : static_cast<float>(k) / (cnt - 1) * inner_h);
      pos[layer_nodes[k]] = {lx, ly};
    }
  }

  // Build node type lookup
  std::unordered_map<std::uint32_t, NodeType> ntype;
  for (const auto &n : nodes)
    ntype[n.id] = n.type;

  // Draw connections
  for (const auto &c : conns) {
    auto it_f = pos.find(c.in_node), it_t = pos.find(c.out_node);
    if (it_f == pos.end() || it_t == pos.end())
      continue;
    sf::VertexArray line(sf::PrimitiveType::Lines, 2);
    sf::Color col = c.enabled ? sf::Color(200, 200, 200, 80) : sf::Color(80, 80, 80, 40);
    line[0].position = it_f->second;
    line[0].color = col;
    line[1].position = it_t->second;
    line[1].color = col;
    target.draw(line);
  }

  // Helper: map activation value [-1, 1] to a color
  // -1 → blue, 0 → gray, +1 → orange
  auto activation_color = [](float val) -> sf::Color {
    val = std::clamp(val, -1.0f, 1.0f);
    if (val < 0.0f) {
      // Interpolate blue → gray
      float t = val + 1.0f; // [0, 1]
      return sf::Color(static_cast<std::uint8_t>(30 + t * (180 - 30)), static_cast<std::uint8_t>(30 + t * (180 - 30)),
                       static_cast<std::uint8_t>(200 + t * (180 - 200)));
    } else {
      // Interpolate gray → orange
      float t = val; // [0, 1]
      return sf::Color(static_cast<std::uint8_t>(180 + t * (220 - 180)),
                       static_cast<std::uint8_t>(180 + t * (120 - 180)),
                       static_cast<std::uint8_t>(180 + t * (20 - 180)));
    }
  };

  // Draw nodes
  sf::CircleShape circle(NODE_R);
  circle.setOutlineThickness(1.0f);
  circle.setOutlineColor(
      sf::Color(ui::NN_NODE_OUTLINE_R, ui::NN_NODE_OUTLINE_G, ui::NN_NODE_OUTLINE_B, ui::NN_NODE_OUTLINE_A));
  for (const auto &n : nodes) {
    auto it = pos.find(n.id);
    if (it == pos.end())
      continue;

    // Use activation color if available, otherwise fall back to type-based
    // color
    auto act_it = node_activations_.find(n.id);
    if (act_it != node_activations_.end()) {
      circle.setFillColor(activation_color(act_it->second));
    } else {
      switch (n.type) {
        case NodeType::Input:
          circle.setFillColor(sf::Color(ui::NN_INPUT_R, ui::NN_INPUT_G, ui::NN_INPUT_B));
          break;
        case NodeType::Bias:
          circle.setFillColor(sf::Color(ui::NN_BIAS_R, ui::NN_BIAS_G, ui::NN_BIAS_B));
          break;
        case NodeType::Hidden:
          circle.setFillColor(sf::Color(ui::NN_HIDDEN_R, ui::NN_HIDDEN_G, ui::NN_HIDDEN_B));
          break;
        case NodeType::Output:
          circle.setFillColor(sf::Color(ui::NN_OUTPUT_R, ui::NN_OUTPUT_G, ui::NN_OUTPUT_B));
          break;
      }
    }
    circle.setPosition({it->second.x - NODE_R, it->second.y - NODE_R});
    target.draw(circle);
  }
}

} // namespace moonai
