#include "visualization/overlay.hpp"

#include "core/types.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <spdlog/spdlog.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace {

constexpr float kNnPanelMargin = 25.0f;
constexpr float kNnPanelGap = 10.0f;
constexpr float kSelectedInfoPanelHeight = 100.0f;
constexpr float kNnPanelTopReserve = 145.0f;
constexpr float kNnPanelMinHeight = 320.0f;
constexpr float kNnPanelMinWidth = 340.0f;
constexpr float kNnPanelMaxWidth = 420.0f;

std::uint8_t lerp_channel(std::uint8_t a, std::uint8_t b, float t) {
  t = std::clamp(t, 0.0f, 1.0f);
  return static_cast<std::uint8_t>(std::lround(static_cast<float>(a) + (static_cast<float>(b) - a) * t));
}

float normalized_rank(int index, int size) {
  if (size <= 1) {
    return 0.5f;
  }
  return static_cast<float>(index) / static_cast<float>(size - 1);
}

float input_visual_slot(std::uint32_t input_index) {
  if (input_index < 10) {
    return static_cast<float>(input_index);
  }
  if (input_index < 20) {
    return 11.5f + static_cast<float>(input_index - 10);
  }
  if (input_index < 30) {
    return 23.0f + static_cast<float>(input_index - 20);
  }
  if (input_index < 33) {
    return 34.5f + static_cast<float>(input_index - 30);
  }
  return 38.5f + static_cast<float>(input_index - 33);
}

float node_visual_slot(const moonai::NodeGene &node) {
  if (node.type == moonai::NodeType::Input) {
    return input_visual_slot(node.id);
  }
  if (node.type == moonai::NodeType::Bias) {
    return 41.5f;
  }
  return 0.0f;
}

sf::Color input_node_color(std::uint32_t input_index) {
  if (input_index < 10) {
    return sf::Color(moonai::chart_colors::PREDATOR_R, moonai::chart_colors::PREDATOR_G,
                     moonai::chart_colors::PREDATOR_B);
  }
  if (input_index < 20) {
    return sf::Color(moonai::chart_colors::PREY_R, moonai::chart_colors::PREY_G, moonai::chart_colors::PREY_B);
  }
  if (input_index < 30) {
    return sf::Color(moonai::chart_colors::FOOD_R, moonai::chart_colors::FOOD_G, moonai::chart_colors::FOOD_B);
  }
  if (input_index < 33) {
    return sf::Color(122, 145, 196);
  }
  return sf::Color(156, 112, 214);
}

sf::Color node_fill_color(const moonai::NodeGene &node) {
  switch (node.type) {
    case moonai::NodeType::Input:
      return input_node_color(node.id);
    case moonai::NodeType::Bias:
      return sf::Color(moonai::ui::NN_BIAS_R, moonai::ui::NN_BIAS_G, moonai::ui::NN_BIAS_B);
    case moonai::NodeType::Hidden:
      return sf::Color(moonai::ui::NN_HIDDEN_R, moonai::ui::NN_HIDDEN_G, moonai::ui::NN_HIDDEN_B);
    case moonai::NodeType::Output:
      return sf::Color(moonai::ui::NN_OUTPUT_R, moonai::ui::NN_OUTPUT_G, moonai::ui::NN_OUTPUT_B);
  }

  return sf::Color::White;
}

sf::Color connection_weight_color(float weight) {
  const float strength = std::clamp(std::abs(weight) / 2.0f, 0.0f, 1.0f);
  const auto alpha = lerp_channel(24, 180, strength);

  if (weight >= 0.0f) {
    return sf::Color(lerp_channel(90, 255, strength), lerp_channel(90, 180, strength), lerp_channel(100, 84, strength),
                     alpha);
  }

  return sf::Color(lerp_channel(90, 82, strength), lerp_channel(90, 188, strength), lerp_channel(100, 255, strength),
                   alpha);
}

struct LayerOrderRef {
  int layer = 0;
  int index = 0;
  int size = 1;
};

std::unordered_map<std::uint32_t, LayerOrderRef>
build_layer_order_lookup(const std::vector<std::vector<std::uint32_t>> &layers) {
  std::unordered_map<std::uint32_t, LayerOrderRef> lookup;
  for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
    const auto &layer_nodes = layers[layer_idx];
    for (std::size_t node_idx = 0; node_idx < layer_nodes.size(); ++node_idx) {
      lookup[layer_nodes[node_idx]] =
          LayerOrderRef{static_cast<int>(layer_idx), static_cast<int>(node_idx), static_cast<int>(layer_nodes.size())};
    }
  }
  return lookup;
}

float average_neighbor_rank(const std::uint32_t node_id,
                            const std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> &neighbors,
                            const std::unordered_map<std::uint32_t, LayerOrderRef> &lookup) {
  const auto neighbor_it = neighbors.find(node_id);
  if (neighbor_it == neighbors.end() || neighbor_it->second.empty()) {
    return 0.5f;
  }

  float rank_sum = 0.0f;
  int count = 0;
  for (const std::uint32_t neighbor_id : neighbor_it->second) {
    const auto order_it = lookup.find(neighbor_id);
    if (order_it == lookup.end()) {
      continue;
    }
    rank_sum += normalized_rank(order_it->second.index, order_it->second.size);
    ++count;
  }

  if (count == 0) {
    return 0.5f;
  }
  return rank_sum / static_cast<float>(count);
}

void reorder_hidden_layers(std::vector<std::vector<std::uint32_t>> &layers,
                           const std::unordered_map<std::uint32_t, const moonai::NodeGene *> &node_by_id,
                           const std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> &incoming,
                           const std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> &outgoing) {
  if (layers.size() <= 2) {
    return;
  }

  for (int pass = 0; pass < 3; ++pass) {
    auto lookup = build_layer_order_lookup(layers);
    for (std::size_t layer_idx = 1; layer_idx + 1 < layers.size(); ++layer_idx) {
      auto &layer_nodes = layers[layer_idx];
      std::stable_sort(layer_nodes.begin(), layer_nodes.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
        if (node_by_id.at(lhs)->type != moonai::NodeType::Hidden ||
            node_by_id.at(rhs)->type != moonai::NodeType::Hidden) {
          return lhs < rhs;
        }

        const float lhs_score = average_neighbor_rank(lhs, incoming, lookup);
        const float rhs_score = average_neighbor_rank(rhs, incoming, lookup);
        if (lhs_score == rhs_score) {
          return lhs < rhs;
        }
        return lhs_score < rhs_score;
      });
    }

    lookup = build_layer_order_lookup(layers);
    for (std::size_t layer_idx = layers.size() - 1; layer_idx > 1; --layer_idx) {
      auto &layer_nodes = layers[layer_idx - 1];
      std::stable_sort(layer_nodes.begin(), layer_nodes.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
        if (node_by_id.at(lhs)->type != moonai::NodeType::Hidden ||
            node_by_id.at(rhs)->type != moonai::NodeType::Hidden) {
          return lhs < rhs;
        }

        const float lhs_score = average_neighbor_rank(lhs, outgoing, lookup);
        const float rhs_score = average_neighbor_rank(rhs, outgoing, lookup);
        if (lhs_score == rhs_score) {
          return lhs < rhs;
        }
        return lhs_score < rhs_score;
      });
    }
  }
}

float compute_layer_visual_span(const std::vector<std::uint32_t> &layer_nodes,
                                const std::unordered_map<std::uint32_t, const moonai::NodeGene *> &node_by_id) {
  if (layer_nodes.empty()) {
    return 1.0f;
  }

  float max_slot = 0.0f;
  bool has_custom_slots = false;
  for (const std::uint32_t node_id : layer_nodes) {
    const auto *node = node_by_id.at(node_id);
    if (node->type == moonai::NodeType::Input || node->type == moonai::NodeType::Bias) {
      max_slot = std::max(max_slot, node_visual_slot(*node));
      has_custom_slots = true;
    }
  }

  if (has_custom_slots) {
    return max_slot + 1.0f;
  }

  return static_cast<float>(layer_nodes.size());
}

std::unordered_set<std::uint32_t> collect_visible_node_ids(const moonai::Genome &genome) {
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> forward_edges;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> reverse_edges;

  for (const auto &conn : genome.connections()) {
    if (!conn.enabled) {
      continue;
    }
    forward_edges[conn.in_node].push_back(conn.out_node);
    reverse_edges[conn.out_node].push_back(conn.in_node);
  }

  std::unordered_set<std::uint32_t> reachable_from_inputs;
  std::queue<std::uint32_t> forward_queue;
  for (const auto &node : genome.nodes()) {
    if (node.type != moonai::NodeType::Input && node.type != moonai::NodeType::Bias) {
      continue;
    }
    reachable_from_inputs.insert(node.id);
    forward_queue.push(node.id);
  }

  while (!forward_queue.empty()) {
    const std::uint32_t node_id = forward_queue.front();
    forward_queue.pop();

    const auto it = forward_edges.find(node_id);
    if (it == forward_edges.end()) {
      continue;
    }

    for (const std::uint32_t next_id : it->second) {
      if (reachable_from_inputs.insert(next_id).second) {
        forward_queue.push(next_id);
      }
    }
  }

  std::unordered_set<std::uint32_t> reaches_output;
  std::queue<std::uint32_t> reverse_queue;
  for (const auto &node : genome.nodes()) {
    if (node.type != moonai::NodeType::Output) {
      continue;
    }
    reaches_output.insert(node.id);
    reverse_queue.push(node.id);
  }

  while (!reverse_queue.empty()) {
    const std::uint32_t node_id = reverse_queue.front();
    reverse_queue.pop();

    const auto it = reverse_edges.find(node_id);
    if (it == reverse_edges.end()) {
      continue;
    }

    for (const std::uint32_t prev_id : it->second) {
      if (reaches_output.insert(prev_id).second) {
        reverse_queue.push(prev_id);
      }
    }
  }

  std::unordered_set<std::uint32_t> visible_nodes;
  for (const auto &node : genome.nodes()) {
    if (node.type == moonai::NodeType::Input || node.type == moonai::NodeType::Bias ||
        node.type == moonai::NodeType::Output) {
      visible_nodes.insert(node.id);
      continue;
    }

    if (reachable_from_inputs.count(node.id) > 0 && reaches_output.count(node.id) > 0) {
      visible_nodes.insert(node.id);
    }
  }

  return visible_nodes;
}

} // namespace

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

  // Bottom-left stacked panels: NN topology and Selected agent info
  float margin = kNnPanelMargin;

  if (selected_genome) {
    draw_nn_panel(target, *selected_genome);
  } else {
    nn_panel_cache_.valid = false;
  }

  if (stats.selected_agent >= 0) {
    float panel_width = 220.0f;
    float line_h = 18.0f;
    char buf[128];
    const float nn_panel_height = nn_panel_cache_.valid ? nn_panel_cache_.panel_height : 0.0f;
    float sel_y =
        target.getDefaultView().getSize().y - margin - nn_panel_height - kNnPanelGap - kSelectedInfoPanelHeight;
    draw_panel(target, margin, sel_y, panel_width, kSelectedInfoPanelHeight);

    float sx = margin + 8.0f;
    float sy = sel_y + 6.0f;

    std::snprintf(buf, sizeof(buf), "Agent #%d", stats.selected_agent);
    draw_text(target, buf, sx, sy, 14, sf::Color(ui::TITLE_R, ui::TITLE_G, ui::TITLE_B));
    sy += line_h;

    std::snprintf(buf, sizeof(buf), "Energy: %.2f  Age: %d", stats.selected_energy, stats.selected_age);
    draw_text(target, buf, sx, sy, 13);
    sy += line_h;

    std::snprintf(buf, sizeof(buf), "Generation: %d", stats.selected_generation);
    draw_text(target, buf, sx, sy, 13);
    sy += line_h;

    std::snprintf(buf, sizeof(buf), "Complexity: %d", stats.selected_genome_complexity);
    draw_text(target, buf, sx, sy, 13, sf::Color(ui::MUTED_R, ui::MUTED_G, ui::MUTED_B));
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

void UIOverlay::push_complexity(float predator_complexity, float prey_complexity) {
  complexity_history_.push_back(std::make_tuple(predator_complexity, prey_complexity));
}

void UIOverlay::push_energy(float predator_energy, float prey_energy) {
  energy_history_.push_back(std::make_tuple(predator_energy, prey_energy));
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

  // Stats widget: Population counts, species, energy, complexity, events, and generation
  draw_stats_widget(target, stats, x, y, PANEL_WIDTH, 400.0f);
  y += 384.0f + MARGIN;

  // Population chart
  draw_population_chart(target, x, y, PANEL_WIDTH, 180.0f);
  y += 180.0f + MARGIN;

  // Complexity chart
  draw_complexity_chart(target, x, y, PANEL_WIDTH, 120.0f);
  y += 120.0f + MARGIN;

  // Energy chart
  draw_energy_chart(target, x, y, PANEL_WIDTH, 120.0f);
  y += 120.0f + MARGIN;

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

  // Population counts
  std::snprintf(buf, sizeof(buf), "Predators: %d", stats.alive_predator);
  draw_text(target, buf, tx, ty, 13,
            sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Prey: %d", stats.alive_prey);
  draw_text(target, buf, tx, ty, 13, sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "Food: %d", stats.active_food);
  draw_text(target, buf, tx, ty, 13, sf::Color(chart_colors::FOOD_R, chart_colors::FOOD_G, chart_colors::FOOD_B));
  ty += line_h + 4;

  // Predator section
  draw_text(target, "Predator:", tx, ty, 13,
            sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Species: %d", stats.predator_species);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Energy: %.2f", stats.avg_predator_energy);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Complx: %.1f", stats.avg_predator_complexity);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Births: %d", stats.total_predator_births);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Deaths: %d", stats.total_predator_deaths);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Gen: max=%d avg=%.1f", stats.max_predator_generation,
                stats.avg_predator_generation);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Kills: %d", stats.total_kills);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h + 4;

  // Prey section
  draw_text(target, "Prey:", tx, ty, 13, sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B));
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Species: %d", stats.prey_species);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Energy: %.2f", stats.avg_prey_energy);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Complx: %.1f", stats.avg_prey_complexity);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Births: %d", stats.total_prey_births);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Deaths: %d", stats.total_prey_deaths);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Gen: max=%d avg=%.1f", stats.max_prey_generation, stats.avg_prey_generation);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
  ty += line_h;

  std::snprintf(buf, sizeof(buf), "  Eaten: %d", stats.total_food_eaten);
  draw_text(target, buf, tx, ty, 13, sf::Color::White);
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

void UIOverlay::draw_complexity_chart(sf::RenderTarget &target, float x, float y, float w, float h) {
  if (!font_loaded_ || complexity_history_.size() < 2)
    return;

  draw_panel(target, x, y, w, h);
  draw_text(target, "Complexity", x + 4.0f, y + 2.0f, 11, sf::Color(180, 180, 200));

  float inner_x = x + 4.0f;
  float inner_y = y + 20.0f;
  float inner_w = w - 8.0f;
  float inner_h = h - 28.0f;

  // Find max complexity for scaling
  float max_complexity = 10.0f;
  for (const auto &t : complexity_history_) {
    max_complexity = std::max(max_complexity, std::max(std::get<0>(t), std::get<1>(t)));
  }

  size_t total_points = complexity_history_.size();

  auto map_point = [&](size_t idx, float val) -> sf::Vector2f {
    float px = inner_x + (static_cast<float>(idx) / (total_points - 1)) * inner_w;
    float py = inner_y + inner_h * (1.0f - val / max_complexity);
    return {px, py};
  };

  // Draw predator complexity line (orange)
  sf::VertexArray pred_line(sf::PrimitiveType::LineStrip, total_points);
  for (size_t i = 0; i < total_points; ++i) {
    pred_line[static_cast<int>(i)].position = map_point(i, std::get<0>(complexity_history_[i]));
    pred_line[static_cast<int>(i)].color =
        sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B);
  }
  target.draw(pred_line);

  // Draw prey complexity line (cyan)
  sf::VertexArray prey_line(sf::PrimitiveType::LineStrip, total_points);
  for (size_t i = 0; i < total_points; ++i) {
    prey_line[static_cast<int>(i)].position = map_point(i, std::get<1>(complexity_history_[i]));
    prey_line[static_cast<int>(i)].color = sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B);
  }
  target.draw(prey_line);

  // Legend
  draw_text(target, "Pred", x + w - 80.0f, y + 4.0f, 10,
            sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B));
  draw_text(target, "Prey", x + w - 40.0f, y + 4.0f, 10,
            sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B));
}

void UIOverlay::draw_energy_chart(sf::RenderTarget &target, float x, float y, float w, float h) {
  if (!font_loaded_ || energy_history_.size() < 2)
    return;

  draw_panel(target, x, y, w, h);
  draw_text(target, "Energy", x + 4.0f, y + 2.0f, 11, sf::Color(180, 180, 200));

  float inner_x = x + 4.0f;
  float inner_y = y + 20.0f;
  float inner_w = w - 8.0f;
  float inner_h = h - 28.0f;

  // Find max energy for scaling
  float max_energy = 2.0f;
  for (const auto &t : energy_history_) {
    max_energy = std::max(max_energy, std::max(std::get<0>(t), std::get<1>(t)));
  }

  size_t total_points = energy_history_.size();

  auto map_point = [&](size_t idx, float val) -> sf::Vector2f {
    float px = inner_x + (static_cast<float>(idx) / (total_points - 1)) * inner_w;
    float py = inner_y + inner_h * (1.0f - val / max_energy);
    return {px, py};
  };

  // Draw predator energy line (orange)
  sf::VertexArray pred_line(sf::PrimitiveType::LineStrip, total_points);
  for (size_t i = 0; i < total_points; ++i) {
    pred_line[static_cast<int>(i)].position = map_point(i, std::get<0>(energy_history_[i]));
    pred_line[static_cast<int>(i)].color =
        sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B);
  }
  target.draw(pred_line);

  // Draw prey energy line (cyan)
  sf::VertexArray prey_line(sf::PrimitiveType::LineStrip, total_points);
  for (size_t i = 0; i < total_points; ++i) {
    prey_line[static_cast<int>(i)].position = map_point(i, std::get<1>(energy_history_[i]));
    prey_line[static_cast<int>(i)].color = sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B);
  }
  target.draw(prey_line);

  // Legend
  draw_text(target, "Pred", x + w - 80.0f, y + 4.0f, 10,
            sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B));
  draw_text(target, "Prey", x + w - 40.0f, y + 4.0f, 10,
            sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B));
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

void UIOverlay::rebuild_nn_panel_cache(const Genome &genome, sf::Vector2f view_size) {
  nn_panel_cache_ = CachedNnPanel{};
  nn_panel_cache_.genome = &genome;
  nn_panel_cache_.view_size = view_size;

  const auto &nodes = genome.nodes();
  const auto &conns = genome.connections();
  const std::unordered_set<std::uint32_t> visible_nodes = collect_visible_node_ids(genome);
  if (visible_nodes.empty()) {
    return;
  }

  std::unordered_map<std::uint32_t, const NodeGene *> node_by_id;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> outgoing;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> incoming;
  std::unordered_map<std::uint32_t, int> remaining_in_degree;
  std::vector<const ConnectionGene *> visible_connections;

  for (const auto &node : nodes) {
    if (visible_nodes.count(node.id) == 0) {
      continue;
    }
    node_by_id[node.id] = &node;
    if (node.type == NodeType::Hidden || node.type == NodeType::Output) {
      remaining_in_degree[node.id] = 0;
    }
  }

  for (const auto &conn : conns) {
    if (!conn.enabled || visible_nodes.count(conn.in_node) == 0 || visible_nodes.count(conn.out_node) == 0) {
      continue;
    }

    outgoing[conn.in_node].push_back(conn.out_node);
    incoming[conn.out_node].push_back(conn.in_node);
    visible_connections.push_back(&conn);
    if (node_by_id.at(conn.out_node)->type == NodeType::Hidden ||
        node_by_id.at(conn.out_node)->type == NodeType::Output) {
      remaining_in_degree[conn.out_node]++;
    }
  }

  std::unordered_map<std::uint32_t, int> depth;
  std::queue<std::uint32_t> ready;
  int max_depth = 0;

  for (const auto &node : nodes) {
    if (visible_nodes.count(node.id) == 0) {
      continue;
    }
    if (node.type == NodeType::Input || node.type == NodeType::Bias) {
      depth[node.id] = 0;
      ready.push(node.id);
    }
  }

  while (!ready.empty()) {
    const std::uint32_t node_id = ready.front();
    ready.pop();

    const auto edge_it = outgoing.find(node_id);
    if (edge_it == outgoing.end()) {
      continue;
    }

    for (const std::uint32_t next_id : edge_it->second) {
      const int next_depth = depth[node_id] + 1;
      const auto depth_it = depth.find(next_id);
      if (depth_it == depth.end() || depth_it->second < next_depth) {
        depth[next_id] = next_depth;
        max_depth = std::max(max_depth, next_depth);
      }

      auto indegree_it = remaining_in_degree.find(next_id);
      if (indegree_it == remaining_in_degree.end()) {
        continue;
      }

      indegree_it->second--;
      if (indegree_it->second == 0 && node_by_id.at(next_id)->type != NodeType::Output) {
        ready.push(next_id);
      }
    }
  }

  for (const auto &node : nodes) {
    if (visible_nodes.count(node.id) == 0 || node.type != NodeType::Hidden || depth.count(node.id) > 0) {
      continue;
    }
    depth[node.id] = 1;
    max_depth = std::max(max_depth, 1);
  }

  for (const auto &node : nodes) {
    if (visible_nodes.count(node.id) > 0 && node.type == NodeType::Output) {
      depth[node.id] = max_depth + 1;
    }
  }

  const int num_layers = max_depth + 2;
  std::vector<std::vector<std::uint32_t>> layers(static_cast<std::size_t>(num_layers));
  for (const auto &node : nodes) {
    if (visible_nodes.count(node.id) == 0) {
      continue;
    }
    layers[static_cast<std::size_t>(depth[node.id])].push_back(node.id);
  }

  if (!layers.empty()) {
    auto &input_layer = layers.front();
    std::stable_sort(input_layer.begin(), input_layer.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
      const auto *left_node = node_by_id.at(lhs);
      const auto *right_node = node_by_id.at(rhs);
      if (left_node->type == NodeType::Bias || right_node->type == NodeType::Bias) {
        return left_node->type != NodeType::Bias;
      }
      return node_visual_slot(*left_node) < node_visual_slot(*right_node);
    });

    auto &output_layer = layers.back();
    std::stable_sort(output_layer.begin(), output_layer.end());
  }

  reorder_hidden_layers(layers, node_by_id, incoming, outgoing);

  float max_visual_span = 1.0f;
  for (const auto &layer_nodes : layers) {
    max_visual_span = std::max(max_visual_span, compute_layer_visual_span(layer_nodes, node_by_id));
  }

  const float available_height =
      std::max(kNnPanelMinHeight,
               view_size.y - kNnPanelTopReserve - kSelectedInfoPanelHeight - kNnPanelGap - (2.0f * kNnPanelMargin));
  const float desired_height = 64.0f + max_visual_span * 12.0f;
  const float panel_height = std::clamp(desired_height, kNnPanelMinHeight, available_height);
  const float panel_width =
      std::clamp(300.0f + static_cast<float>(num_layers) * 18.0f, kNnPanelMinWidth, kNnPanelMaxWidth);
  const float panel_x = kNnPanelMargin;
  const float panel_y = view_size.y - kNnPanelMargin - panel_height;

  const float inner_x = panel_x + 16.0f;
  const float inner_y = panel_y + 24.0f;
  const float inner_w = panel_width - 32.0f;
  const float inner_h = panel_height - 40.0f;
  const float node_radius = std::clamp(inner_h / (max_visual_span * 2.8f), 2.6f, 5.0f);
  const float top_padding = node_radius + 3.0f;
  const float usable_h = std::max(1.0f, inner_h - (2.0f * top_padding));

  std::unordered_map<std::uint32_t, sf::Vector2f> positions;
  for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
    const auto &layer_nodes = layers[layer_idx];
    if (layer_nodes.empty()) {
      continue;
    }

    const float layer_x = inner_x + (layers.size() <= 1 ? inner_w * 0.5f
                                                        : static_cast<float>(layer_idx) /
                                                              static_cast<float>(layers.size() - 1) * inner_w);

    if (layer_idx == 0) {
      float max_slot = 0.0f;
      for (const std::uint32_t node_id : layer_nodes) {
        max_slot = std::max(max_slot, node_visual_slot(*node_by_id.at(node_id)));
      }

      if (max_slot <= 0.0f) {
        positions[layer_nodes.front()] = {layer_x, inner_y + inner_h * 0.5f};
      } else {
        for (const std::uint32_t node_id : layer_nodes) {
          const float slot = node_visual_slot(*node_by_id.at(node_id));
          const float y = inner_y + top_padding + (slot / max_slot) * usable_h;
          positions[node_id] = {layer_x, y};
        }
      }
      continue;
    }

    const int count = static_cast<int>(layer_nodes.size());
    for (int node_idx = 0; node_idx < count; ++node_idx) {
      const float y = inner_y + top_padding + normalized_rank(node_idx, count) * usable_h;
      positions[layer_nodes[static_cast<std::size_t>(node_idx)]] = {layer_x, y};
    }
  }

  nn_panel_cache_.panel_x = panel_x;
  nn_panel_cache_.panel_y = panel_y;
  nn_panel_cache_.panel_width = panel_width;
  nn_panel_cache_.panel_height = panel_height;
  nn_panel_cache_.node_radius = node_radius;
  nn_panel_cache_.connection_lines.setPrimitiveType(sf::PrimitiveType::Lines);
  nn_panel_cache_.connection_lines.clear();
  nn_panel_cache_.connection_lines.resize(visible_connections.size() * 2);

  std::size_t vertex_idx = 0;
  for (const ConnectionGene *const conn : visible_connections) {
    const auto from_it = positions.find(conn->in_node);
    const auto to_it = positions.find(conn->out_node);
    if (from_it == positions.end() || to_it == positions.end()) {
      continue;
    }

    const sf::Color color = connection_weight_color(conn->weight);
    nn_panel_cache_.connection_lines[vertex_idx] = sf::Vertex{from_it->second, color};
    nn_panel_cache_.connection_lines[vertex_idx + 1] = sf::Vertex{to_it->second, color};
    vertex_idx += 2;
  }
  nn_panel_cache_.connection_lines.resize(vertex_idx);

  nn_panel_cache_.nodes.clear();
  nn_panel_cache_.nodes.reserve(visible_nodes.size());
  for (const auto &node : nodes) {
    if (visible_nodes.count(node.id) == 0) {
      continue;
    }

    const auto pos_it = positions.find(node.id);
    if (pos_it == positions.end()) {
      continue;
    }

    nn_panel_cache_.nodes.push_back(CachedNnNode{node.id, node.type, pos_it->second, node_fill_color(node)});
  }

  nn_panel_cache_.valid = true;
}

void UIOverlay::draw_nn_panel(sf::RenderTarget &target, const Genome &genome) {
  const sf::Vector2f view_size = target.getDefaultView().getSize();
  if (!nn_panel_cache_.valid || nn_panel_cache_.genome != &genome || nn_panel_cache_.view_size != view_size) {
    rebuild_nn_panel_cache(genome, view_size);
  }

  if (!nn_panel_cache_.valid) {
    return;
  }

  draw_panel(target, nn_panel_cache_.panel_x, nn_panel_cache_.panel_y, nn_panel_cache_.panel_width,
             nn_panel_cache_.panel_height);
  draw_text(target, "Network", nn_panel_cache_.panel_x + 4.0f, nn_panel_cache_.panel_y + 2.0f, 11,
            sf::Color(ui::TITLE_R, ui::TITLE_G, ui::TITLE_B));

  target.draw(nn_panel_cache_.connection_lines);

  sf::CircleShape circle(nn_panel_cache_.node_radius);
  circle.setOutlineThickness(1.0f);
  circle.setOutlineColor(
      sf::Color(ui::NN_NODE_OUTLINE_R, ui::NN_NODE_OUTLINE_G, ui::NN_NODE_OUTLINE_B, ui::NN_NODE_OUTLINE_A));

  for (const auto &node : nn_panel_cache_.nodes) {
    circle.setFillColor(node.color);
    circle.setPosition({node.position.x - nn_panel_cache_.node_radius, node.position.y - nn_panel_cache_.node_radius});
    target.draw(circle);
  }
}

} // namespace moonai
