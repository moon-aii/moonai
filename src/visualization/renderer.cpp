#include "visualization/renderer.hpp"
#include "visualization/constants.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <cmath>

namespace moonai {

Renderer::Renderer() {
  initialize_circle_template();
}

void Renderer::initialize_circle_template() {
  constexpr float pi = 3.14159265358979323846f;
  const float step = 2.0f * pi / static_cast<float>(kPreyCircleSegments);

  for (int i = 0; i <= kPreyCircleSegments; ++i) {
    const float angle = step * static_cast<float>(i);
    prey_circle_template_[i] = Vec2{std::cos(angle), std::sin(angle)};
  }
}

sf::Color Renderer::brighten_color(sf::Color color, int amount) {
  color.r = static_cast<std::uint8_t>(std::min(255, color.r + amount));
  color.g = static_cast<std::uint8_t>(std::min(255, color.g + amount));
  color.b = static_cast<std::uint8_t>(std::min(255, color.b + amount));
  return color;
}

void Renderer::draw_triangles(sf::RenderTarget &target, const std::vector<sf::Vertex> &vertices) const {
  if (vertices.empty()) {
    return;
  }

  target.draw(vertices.data(), vertices.size(), sf::PrimitiveType::Triangles);
}

void Renderer::draw_background(sf::RenderTarget &target, int width, int height) {
  rect_.setSize({static_cast<float>(width), static_cast<float>(height)});
  rect_.setPosition({0.0f, 0.0f});
  rect_.setFillColor(sf::Color(visual::BG_R, visual::BG_G, visual::BG_B));
  target.draw(rect_);
}

void Renderer::draw_grid(sf::RenderTarget &target, int width, int height, float cell_size) {
  sf::VertexArray lines(sf::PrimitiveType::Lines);
  sf::Color grid_color(visual::GRID_R, visual::GRID_G, visual::GRID_B);

  for (float x = 0; x <= static_cast<float>(width); x += cell_size) {
    lines.append(sf::Vertex{{x, 0.0f}, grid_color});
    lines.append(sf::Vertex{{x, static_cast<float>(height)}, grid_color});
  }
  for (float y = 0; y <= static_cast<float>(height); y += cell_size) {
    lines.append(sf::Vertex{{0.0f, y}, grid_color});
    lines.append(sf::Vertex{{static_cast<float>(width), y}, grid_color});
  }

  target.draw(lines);
}

void Renderer::draw_boundaries(sf::RenderTarget &target, int width, int height) {
  sf::VertexArray border(sf::PrimitiveType::LineStrip, 5);
  sf::Color border_color(visual::BORDER_R, visual::BORDER_G, visual::BORDER_B);

  float w = static_cast<float>(width);
  float h = static_cast<float>(height);
  border[0] = sf::Vertex{{0, 0}, border_color};
  border[1] = sf::Vertex{{w, 0}, border_color};
  border[2] = sf::Vertex{{w, h}, border_color};
  border[3] = sf::Vertex{{0, h}, border_color};
  border[4] = sf::Vertex{{0, 0}, border_color};

  target.draw(border);
}

void Renderer::draw_food(sf::RenderTarget &target, const std::vector<RenderFood> &food) {
  sf::Color color(chart_colors::FOOD_R, chart_colors::FOOD_G, chart_colors::FOOD_B, visual::FOOD_ALPHA);
  food_vertices_.resize(food.size() * 6);

  std::size_t vertex_index = 0;
  for (const auto &item : food) {
    const float r = sizes::FOOD_RADIUS;
    write_quad(food_vertices_.data() + vertex_index, item.position, r, color);
    vertex_index += 6;
  }

  draw_triangles(target, food_vertices_);
}

void Renderer::write_quad(sf::Vertex *vertices, Vec2 position, float half_extent, sf::Color color) {
  const float left = position.x - half_extent;
  const float right = position.x + half_extent;
  const float top = position.y - half_extent;
  const float bottom = position.y + half_extent;

  vertices[0] = sf::Vertex{{left, top}, color};
  vertices[1] = sf::Vertex{{right, top}, color};
  vertices[2] = sf::Vertex{{right, bottom}, color};
  vertices[3] = sf::Vertex{{left, top}, color};
  vertices[4] = sf::Vertex{{right, bottom}, color};
  vertices[5] = sf::Vertex{{left, bottom}, color};
}

void Renderer::write_triangle(sf::Vertex *vertices, const RenderAgent &agent, float size, sf::Color color) {
  Vec2 forward = agent.velocity;
  const float velocity_length_sq = forward.x * forward.x + forward.y * forward.y;
  if (velocity_length_sq > 1e-6f) {
    const float inv_length = 1.0f / std::sqrt(velocity_length_sq);
    forward.x *= inv_length;
    forward.y *= inv_length;
  } else {
    forward = Vec2{1.0f, 0.0f};
  }

  const Vec2 perpendicular{-forward.y, forward.x};
  const float tip_scale = size * visual::TRIANGLE_TIP_FACTOR;
  const float base_scale = size * visual::TRIANGLE_BASE_FACTOR;
  const float width_scale = size * visual::TRIANGLE_WIDTH_FACTOR;
  const Vec2 tip{agent.position.x + forward.x * tip_scale, agent.position.y + forward.y * tip_scale};
  const Vec2 left{agent.position.x - forward.x * base_scale - perpendicular.x * width_scale,
                  agent.position.y - forward.y * base_scale - perpendicular.y * width_scale};
  const Vec2 right{agent.position.x - forward.x * base_scale + perpendicular.x * width_scale,
                   agent.position.y - forward.y * base_scale + perpendicular.y * width_scale};

  vertices[0] = sf::Vertex{{tip.x, tip.y}, color};
  vertices[1] = sf::Vertex{{left.x, left.y}, color};
  vertices[2] = sf::Vertex{{right.x, right.y}, color};
}

void Renderer::write_circle(sf::Vertex *vertices, Vec2 position, float radius, sf::Color color) const {
  for (int i = 0; i < kPreyCircleSegments; ++i) {
    const Vec2 &point_a = prey_circle_template_[i];
    const Vec2 &point_b = prey_circle_template_[i + 1];
    const int base = i * 3;

    vertices[base] = sf::Vertex{{position.x, position.y}, color};
    vertices[base + 1] = sf::Vertex{{position.x + radius * point_a.x, position.y + radius * point_a.y}, color};
    vertices[base + 2] = sf::Vertex{{position.x + radius * point_b.x, position.y + radius * point_b.y}, color};
  }
}

void Renderer::draw_predators(sf::RenderTarget &target, const std::vector<RenderAgent> &predators,
                              uint32_t selected_agent_id) {
  bool has_selected = false;
  RenderAgent selected_agent;

  predator_vertices_.resize(predators.size() * 3);

  std::size_t predator_index = 0;
  const sf::Color predator_color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B);

  for (const auto &agent : predators) {
    if (selected_agent_id != 0 && agent.agent_id == selected_agent_id) {
      has_selected = true;
      selected_agent = agent;
    }

    write_triangle(predator_vertices_.data() + predator_index, agent, sizes::PREDATOR_RADIUS, predator_color);
    predator_index += 3;
  }

  draw_triangles(target, predator_vertices_);

  if (has_selected) {
    const sf::Color selected_fill = brighten_color(predator_color, 60);
    const sf::Color outline_color = brighten_color(predator_color, 30);
    selected_vertices_.resize(6);
    write_triangle(selected_vertices_.data(), selected_agent,
                   sizes::PREDATOR_RADIUS + visual::SELECTED_OUTLINE_THICKNESS, outline_color);
    write_triangle(selected_vertices_.data() + 3, selected_agent, sizes::PREDATOR_RADIUS, selected_fill);
    draw_triangles(target, selected_vertices_);
  }
}

void Renderer::draw_prey(sf::RenderTarget &target, const std::vector<RenderAgent> &prey, uint32_t selected_agent_id) {
  bool has_selected = false;
  RenderAgent selected_agent;

  prey_vertices_.resize(prey.size() * kPreyCircleSegments * 3);
  std::size_t prey_index = 0;
  const sf::Color prey_color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B);

  for (const auto &agent : prey) {
    if (selected_agent_id != 0 && agent.agent_id == selected_agent_id) {
      has_selected = true;
      selected_agent = agent;
    }

    write_circle(prey_vertices_.data() + prey_index, agent.position, sizes::PREY_RADIUS, prey_color);
    prey_index += kPreyCircleSegments * 3;
  }

  draw_triangles(target, prey_vertices_);

  if (has_selected) {
    const sf::Color selected_fill = brighten_color(prey_color, 60);
    const sf::Color outline_color = brighten_color(prey_color, 30);
    selected_vertices_.resize(kPreyCircleSegments * 6);
    write_circle(selected_vertices_.data(), selected_agent.position,
                 sizes::PREY_RADIUS + visual::SELECTED_OUTLINE_THICKNESS, outline_color);
    write_circle(selected_vertices_.data() + (kPreyCircleSegments * 3), selected_agent.position, sizes::PREY_RADIUS,
                 selected_fill);
    draw_triangles(target, selected_vertices_);
  }
}

void Renderer::draw_vision_range(sf::RenderTarget &target, Vec2 position, float vision_range) {
  sf::CircleShape vision(vision_range, visual::VISION_POINT_COUNT);
  vision.setOrigin({vision_range, vision_range});
  vision.setPosition({position.x, position.y});
  vision.setFillColor(
      sf::Color(visual::VISION_FILL_R, visual::VISION_FILL_G, visual::VISION_FILL_B, visual::VISION_FILL_ALPHA));
  vision.setOutlineColor(
      sf::Color(visual::VISION_FILL_R, visual::VISION_FILL_G, visual::VISION_FILL_B, visual::VISION_OUTLINE_ALPHA));
  vision.setOutlineThickness(1.0f);
  target.draw(vision);
}

void Renderer::draw_sensor_lines(sf::RenderTarget &target, const std::vector<RenderLine> &lines_in) {
  sf::VertexArray lines(sf::PrimitiveType::Lines);
  for (const auto &line : lines_in) {
    lines.append(sf::Vertex{{line.start.x, line.start.y}, line.color});
    lines.append(sf::Vertex{{line.end.x, line.end.y}, line.color});
  }

  target.draw(lines);
}

} // namespace moonai
