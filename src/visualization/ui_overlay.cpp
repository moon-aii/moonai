#include "visualization/ui_overlay.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <spdlog/spdlog.h>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <queue>
#include <unordered_map>

namespace moonai {

bool UIOverlay::initialize(const std::string& font_path) {
    // Try specified path first, then common system font locations
    std::vector<std::string> paths;
    if (!font_path.empty()) {
        paths.push_back(font_path);
    }

    // Common font locations
    paths.push_back("/usr/share/fonts/TTF/DejaVuSansMono.ttf");
    paths.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf");
    paths.push_back("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf");
    paths.push_back("/usr/share/fonts/TTF/DejaVuSans.ttf");
    paths.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    paths.push_back("/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf");
    paths.push_back("/usr/share/fonts/liberation/LiberationMono-Regular.ttf");
    paths.push_back("/usr/share/fonts/noto/NotoSansMono-Regular.ttf");
    paths.push_back("/usr/share/fonts/noto-cjk/NotoSansMono-Regular.ttf");
    paths.push_back("C:\\Windows\\Fonts\\consola.ttf");
    paths.push_back("C:\\Windows\\Fonts\\cour.ttf");

    for (const auto& p : paths) {
        if (std::filesystem::exists(p) && font_.openFromFile(p)) {
            font_loaded_ = true;
            spdlog::debug("Loaded font: {}", p);
            return true;
        }
    }

    spdlog::warn("No system font found. UI overlay will be disabled.");
    font_loaded_ = false;
    return false;
}

void UIOverlay::draw(sf::RenderTarget& target, const OverlayStats& stats,
                     const Genome* selected_genome) {
    if (!font_loaded_) return;

    // Get the current view to draw UI in screen space
    sf::View ui_view = target.getDefaultView();
    sf::View current_view = target.getView();
    target.setView(ui_view);

    float panel_width = 220.0f;
    float panel_height = 210.0f;
    float margin = 10.0f;

    // Top-left: simulation stats panel
    draw_panel(target, margin, margin, panel_width, panel_height);

    float x = margin + 8.0f;
    float y = margin + 6.0f;
    float line_h = 18.0f;

    if (!stats.experiment_name.empty()) {
        draw_text(target, stats.experiment_name, x, y, 16, sf::Color(200, 200, 255));
    } else {
        draw_text(target, "MoonAI", x, y, 16, sf::Color(200, 200, 255));
    }
    y += line_h + 4;

    char buf[128];

    std::snprintf(buf, sizeof(buf), "Gen: %d  Tick: %d%s",
                  stats.generation, stats.tick,
                  stats.fast_forward ? "  [FF]" : "");
    draw_text(target, buf, x, y, 13,
              stats.fast_forward ? sf::Color(100, 220, 255) : sf::Color::White);
    y += line_h;

    std::snprintf(buf, sizeof(buf), "FPS: %.0f", stats.fps);
    draw_text(target, buf, x, y, 13, sf::Color(180, 180, 180));
    y += line_h;

    std::snprintf(buf, sizeof(buf), "Predators: %d", stats.alive_predators);
    draw_text(target, buf, x, y, 13, sf::Color(220, 80, 80));
    y += line_h;

    std::snprintf(buf, sizeof(buf), "Prey: %d", stats.alive_prey);
    draw_text(target, buf, x, y, 13, sf::Color(80, 220, 100));
    y += line_h;

    std::snprintf(buf, sizeof(buf), "Species: %d", stats.num_species);
    draw_text(target, buf, x, y, 13);
    y += line_h;

    std::snprintf(buf, sizeof(buf), "Best: %.2f  Avg: %.2f",
                  stats.best_fitness, stats.avg_fitness);
    draw_text(target, buf, x, y, 13, sf::Color(255, 220, 100));
    y += line_h;

    std::snprintf(buf, sizeof(buf), "Speed: %dx%s",
                  stats.speed_multiplier,
                  stats.paused ? " [PAUSED]" : "");
    draw_text(target, buf, x, y, 13,
              stats.paused ? sf::Color(255, 150, 100) : sf::Color(180, 180, 180));
    y += line_h + 4;

    // Controls hint
    draw_text(target, "[Space] Pause  [+/-] Speed", x, y, 11, sf::Color(120, 120, 140));
    y += 14;
    draw_text(target, "[G] Grid  [V] Vision  [H] FF  [E] Exp  [R] Reset", x, y, 11, sf::Color(120, 120, 140));

    // Selected agent panel (bottom-left)
    if (stats.selected_agent >= 0) {
        float sel_y = target.getDefaultView().getSize().y - margin - 130.0f;
        draw_panel(target, margin, sel_y, panel_width, 120.0f);

        float sx = margin + 8.0f;
        float sy = sel_y + 6.0f;

        std::snprintf(buf, sizeof(buf), "Agent #%d", stats.selected_agent);
        draw_text(target, buf, sx, sy, 14, sf::Color(200, 200, 255));
        sy += line_h;

        std::snprintf(buf, sizeof(buf), "Energy: %.1f  Age: %d",
                      stats.selected_energy, stats.selected_age);
        draw_text(target, buf, sx, sy, 13);
        sy += line_h;

        std::snprintf(buf, sizeof(buf), "Kills: %d  Food: %d",
                      stats.selected_kills, stats.selected_food_eaten);
        draw_text(target, buf, sx, sy, 13);
        sy += line_h;

        std::snprintf(buf, sizeof(buf), "Fitness: %.2f", stats.selected_fitness);
        draw_text(target, buf, sx, sy, 13, sf::Color(255, 220, 100));
        sy += line_h;

        std::snprintf(buf, sizeof(buf), "Complexity: %d", stats.selected_genome_complexity);
        draw_text(target, buf, sx, sy, 13, sf::Color(180, 180, 180));
    }

    // Real-time fitness chart (bottom-right)
    if (best_history_.size() >= 2) {
        draw_fitness_chart(target);
    }

    // NN topology panel (above fitness chart, anchored to right edge)
    if (selected_genome) {
        draw_nn_panel(target, *selected_genome);
    }

    // Restore the camera view
    target.setView(current_view);
}

void UIOverlay::draw_panel(sf::RenderTarget& target, float x, float y, float w, float h) {
    panel_bg_.setSize({w, h});
    panel_bg_.setPosition({x, y});
    panel_bg_.setFillColor(sf::Color(10, 10, 20, 200));
    panel_bg_.setOutlineColor(sf::Color(60, 60, 80, 200));
    panel_bg_.setOutlineThickness(1.0f);
    target.draw(panel_bg_);
}

void UIOverlay::draw_text(sf::RenderTarget& target, const std::string& str,
                           float x, float y, unsigned int size, sf::Color color) {
    sf::Text text(font_, str, size);
    text.setPosition({x, y});
    text.setFillColor(color);
    target.draw(text);
}

void UIOverlay::push_fitness(float best, float avg) {
    best_history_.push_back(best);
    avg_history_.push_back(avg);
    if (static_cast<int>(best_history_.size()) > CHART_MAX_POINTS) {
        best_history_.pop_front();
        avg_history_.pop_front();
    }
}

void UIOverlay::set_activations(const std::unordered_map<std::uint32_t, float>& vals) {
    node_activations_ = vals;
}

void UIOverlay::draw_fitness_chart(sf::RenderTarget& target) {
    if (!font_loaded_) return;

    float chart_w = 300.0f;
    float chart_h = 100.0f;
    float margin = 10.0f;
    sf::Vector2f view_size = target.getDefaultView().getSize();
    float cx = view_size.x - chart_w - margin;
    float cy = view_size.y - chart_h - margin;

    draw_panel(target, cx, cy, chart_w, chart_h);

    // Find max value for scaling
    float max_val = 0.1f;
    for (float v : best_history_) max_val = std::max(max_val, v);
    for (float v : avg_history_)  max_val = std::max(max_val, v);

    int n = static_cast<int>(best_history_.size());
    float inner_x = cx + 4.0f;
    float inner_y = cy + 4.0f;
    float inner_w = chart_w - 8.0f;
    float inner_h = chart_h - 16.0f;

    auto map_point = [&](int idx, float val) -> sf::Vector2f {
        float px = inner_x + (static_cast<float>(idx) / (n - 1)) * inner_w;
        float py = inner_y + inner_h * (1.0f - val / max_val);
        return {px, py};
    };

    // Draw best fitness line (blue)
    sf::VertexArray best_line(sf::PrimitiveType::LineStrip, n);
    for (int i = 0; i < n; ++i) {
        best_line[i].position = map_point(i, best_history_[i]);
        best_line[i].color = sf::Color(100, 150, 255);
    }
    target.draw(best_line);

    // Draw avg fitness line (green)
    sf::VertexArray avg_line(sf::PrimitiveType::LineStrip, n);
    for (int i = 0; i < n; ++i) {
        avg_line[i].position = map_point(i, avg_history_[i]);
        avg_line[i].color = sf::Color(100, 220, 100);
    }
    target.draw(avg_line);

    // Labels
    draw_text(target, "Fitness", cx + 4.0f, cy + chart_h - 14.0f, 11,
              sf::Color(180, 180, 180));
}

void UIOverlay::draw_nn_panel(sf::RenderTarget& target, const Genome& genome) {
    constexpr float PANEL_W = 250.0f;
    constexpr float PANEL_H = 300.0f;
    constexpr float MARGIN  = 10.0f;
    constexpr float NODE_R  = 5.0f;

    sf::Vector2f view_size = target.getDefaultView().getSize();

    // Position: above the fitness chart (chart is 100px + margin at the bottom)
    float chart_h = (best_history_.size() >= 2) ? 110.0f : 0.0f;
    float cx = view_size.x - PANEL_W - MARGIN;
    float cy = view_size.y - PANEL_H - MARGIN - chart_h;

    draw_panel(target, cx, cy, PANEL_W, PANEL_H);
    draw_text(target, "Network", cx + 4.0f, cy + 2.0f, 11, sf::Color(180, 180, 200));

    const auto& nodes = genome.nodes();
    const auto& conns = genome.connections();

    // Assign layer depth via BFS from input/bias nodes
    std::unordered_map<std::uint32_t, int> depth;
    int max_depth = 0;
    std::queue<std::uint32_t> bfs_queue;

    for (const auto& n : nodes) {
        if (n.type == NodeType::Input || n.type == NodeType::Bias) {
            depth[n.id] = 0;
            bfs_queue.push(n.id);
        } else if (n.type == NodeType::Output) {
            depth[n.id] = 2;  // outputs start at depth 2; will be pushed by hidden if deeper
        }
    }
    while (!bfs_queue.empty()) {
        auto nid = bfs_queue.front(); bfs_queue.pop();
        for (const auto& c : conns) {
            if (!c.enabled || c.in_node != nid) continue;
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
    for (const auto& n : nodes) {
        if (n.type == NodeType::Output) depth[n.id] = max_depth + 1;
    }
    int num_layers = max_depth + 2;

    // Group nodes by layer
    std::unordered_map<int, std::vector<std::uint32_t>> layers;
    for (const auto& n : nodes) {
        layers[depth[n.id]].push_back(n.id);
    }

    // Compute pixel position per node
    float inner_x = cx + 10.0f;
    float inner_y = cy + 20.0f;
    float inner_w = PANEL_W - 20.0f;
    float inner_h = PANEL_H - 28.0f;

    std::unordered_map<std::uint32_t, sf::Vector2f> pos;
    for (auto& [layer, layer_nodes] : layers) {
        float lx = inner_x + (num_layers <= 1 ? inner_w / 2.0f
            : static_cast<float>(layer) / (num_layers - 1) * inner_w);
        int cnt = static_cast<int>(layer_nodes.size());
        for (int k = 0; k < cnt; ++k) {
            float ly = inner_y + (cnt <= 1 ? inner_h / 2.0f
                : static_cast<float>(k) / (cnt - 1) * inner_h);
            pos[layer_nodes[k]] = {lx, ly};
        }
    }

    // Build node type lookup
    std::unordered_map<std::uint32_t, NodeType> ntype;
    for (const auto& n : nodes) ntype[n.id] = n.type;

    // Draw connections
    for (const auto& c : conns) {
        auto it_f = pos.find(c.in_node), it_t = pos.find(c.out_node);
        if (it_f == pos.end() || it_t == pos.end()) continue;
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
            float t = val + 1.0f;  // [0, 1]
            return sf::Color(
                static_cast<std::uint8_t>(30 + t * (180 - 30)),
                static_cast<std::uint8_t>(30 + t * (180 - 30)),
                static_cast<std::uint8_t>(200 + t * (180 - 200)));
        } else {
            // Interpolate gray → orange
            float t = val;  // [0, 1]
            return sf::Color(
                static_cast<std::uint8_t>(180 + t * (220 - 180)),
                static_cast<std::uint8_t>(180 + t * (120 - 180)),
                static_cast<std::uint8_t>(180 + t * (20 - 180)));
        }
    };

    // Draw nodes
    sf::CircleShape circle(NODE_R);
    circle.setOutlineThickness(1.0f);
    circle.setOutlineColor(sf::Color(200, 200, 200, 120));
    for (const auto& n : nodes) {
        auto it = pos.find(n.id);
        if (it == pos.end()) continue;

        // Use activation color if available, otherwise fall back to type-based color
        auto act_it = node_activations_.find(n.id);
        if (act_it != node_activations_.end()) {
            circle.setFillColor(activation_color(act_it->second));
        } else {
            switch (n.type) {
                case NodeType::Input: circle.setFillColor(sf::Color(80, 120, 220)); break;
                case NodeType::Bias:  circle.setFillColor(sf::Color(60, 180, 220)); break;
                case NodeType::Hidden: circle.setFillColor(sf::Color(220, 200, 80)); break;
                case NodeType::Output: circle.setFillColor(sf::Color(220, 80, 80)); break;
            }
        }
        circle.setPosition({it->second.x - NODE_R, it->second.y - NODE_R});
        target.draw(circle);
    }
}

int UIOverlay::draw_experiment_selector(sf::RenderTarget& target,
                                        const std::vector<std::string>& names,
                                        int hover_index, int scroll_offset) {
    if (!font_loaded_ || names.empty()) return -1;

    sf::View ui_view = target.getDefaultView();
    sf::View current_view = target.getView();
    target.setView(ui_view);

    sf::Vector2f view_size = ui_view.getSize();

    // Dark fullscreen backdrop
    sf::RectangleShape backdrop;
    backdrop.setSize(view_size);
    backdrop.setPosition({0.0f, 0.0f});
    backdrop.setFillColor(sf::Color(0, 0, 0, 180));
    target.draw(backdrop);

    // Panel dimensions
    float panel_w = 400.0f;
    float panel_h = std::min(view_size.y - 80.0f, 500.0f);
    float panel_x = (view_size.x - panel_w) / 2.0f;
    float panel_y = (view_size.y - panel_h) / 2.0f;

    draw_panel(target, panel_x, panel_y, panel_w, panel_h);

    // Title
    draw_text(target, "Select Experiment", panel_x + 12.0f, panel_y + 10.0f, 18,
              sf::Color(200, 200, 255));
    draw_text(target, "Click to select, Scroll to navigate, ESC to cancel",
              panel_x + 12.0f, panel_y + 34.0f, 11, sf::Color(120, 120, 140));

    // List area
    float list_y = panel_y + 56.0f;
    float list_h = panel_h - 66.0f;
    float item_h = 28.0f;
    int visible_count = static_cast<int>(list_h / item_h);
    int total = static_cast<int>(names.size());

    int clicked = -1;

    for (int i = 0; i < visible_count && (i + scroll_offset) < total; ++i) {
        int idx = i + scroll_offset;
        float iy = list_y + i * item_h;

        // Hover highlight
        if (idx == hover_index) {
            sf::RectangleShape highlight;
            highlight.setSize({panel_w - 16.0f, item_h - 2.0f});
            highlight.setPosition({panel_x + 8.0f, iy});
            highlight.setFillColor(sf::Color(60, 60, 100, 150));
            target.draw(highlight);
        }

        sf::Color text_color = (idx == hover_index)
            ? sf::Color(255, 220, 100)
            : sf::Color(200, 200, 200);

        draw_text(target, names[idx], panel_x + 16.0f, iy + 4.0f, 14, text_color);
    }

    // Scroll indicator
    if (total > visible_count) {
        char scroll_buf[64];
        std::snprintf(scroll_buf, sizeof(scroll_buf), "[%d-%d of %d]",
                      scroll_offset + 1,
                      std::min(scroll_offset + visible_count, total), total);
        draw_text(target, scroll_buf, panel_x + panel_w - 120.0f, panel_y + 10.0f,
                  11, sf::Color(120, 120, 140));
    }

    target.setView(current_view);
    return clicked;
}

} // namespace moonai
