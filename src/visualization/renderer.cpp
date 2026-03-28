#include "visualization/renderer.hpp"
#include "simulation/components.hpp"
#include "visualization/visual_constants.hpp"

#include <SFML/Graphics/PrimitiveType.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <cmath>

namespace moonai {

Renderer::Renderer() {
  // Pre-configure triangle shape (3 points for predator)
  triangle_.setPointCount(3);

  // Pre-configure circle for prey
  circle_.setPointCount(20);
}

void Renderer::draw_background(sf::RenderTarget &target, int width,
                               int height) {
  rect_.setSize({static_cast<float>(width), static_cast<float>(height)});
  rect_.setPosition({0.0f, 0.0f});
  rect_.setFillColor(sf::Color(visual::BG_R, visual::BG_G, visual::BG_B));
  target.draw(rect_);
}

void Renderer::draw_grid(sf::RenderTarget &target, int width, int height,
                         float cell_size) {
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

void Renderer::draw_boundaries(sf::RenderTarget &target, int width,
                               int height) {
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

void Renderer::draw_food(sf::RenderTarget &target,
                         const std::vector<RenderFood> &food) {
  circle_.setRadius(sizes::FOOD_RADIUS);
  circle_.setOrigin({sizes::FOOD_RADIUS, sizes::FOOD_RADIUS});

  sf::Color color(chart_colors::FOOD_R, chart_colors::FOOD_G,
                  chart_colors::FOOD_B, visual::FOOD_ALPHA);

  for (const auto &item : food) {
    circle_.setPosition({item.position.x, item.position.y});
    circle_.setFillColor(color);
    circle_.setOutlineThickness(0);
    target.draw(circle_);
  }
}

void Renderer::draw_agent(sf::RenderTarget &target, const RenderAgent &agent,
                          bool selected) {
  sf::Color base_color;
  float visual_radius;
  if (agent.type == IdentitySoA::TYPE_PREDATOR) {
    base_color = sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G,
                           chart_colors::PREDATOR_B);
    visual_radius = sizes::PREDATOR_RADIUS;
  } else {
    base_color = sf::Color(chart_colors::PREY_R, chart_colors::PREY_G,
                           chart_colors::PREY_B);
    visual_radius = sizes::PREY_RADIUS;
  }

  if (selected) {
    base_color.r = std::min(255, base_color.r + 60);
    base_color.g = std::min(255, base_color.g + 60);
    base_color.b = std::min(255, base_color.b + 60);
  }

  sf::Color outline_color(std::min(255, base_color.r + 30),
                          std::min(255, base_color.g + 30),
                          std::min(255, base_color.b + 30));

  bool is_triangle = agent.type == IdentitySoA::TYPE_PREDATOR;

  if (is_triangle) {
    float size = visual_radius;
    Vec2 vel = agent.velocity;
    float angle = std::atan2(vel.y, vel.x);

    float x = agent.position.x;
    float y = agent.position.y;

    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);

    triangle_.setPoint(0, {x + cos_a * size * visual::TRIANGLE_TIP_FACTOR,
                           y + sin_a * size * visual::TRIANGLE_TIP_FACTOR});
    float lx = -size * visual::TRIANGLE_BASE_FACTOR,
          ly = -size * visual::TRIANGLE_WIDTH_FACTOR;
    triangle_.setPoint(
        1, {x + cos_a * lx - sin_a * ly, y + sin_a * lx + cos_a * ly});
    float rx = -size * visual::TRIANGLE_BASE_FACTOR,
          ry = size * visual::TRIANGLE_WIDTH_FACTOR;
    triangle_.setPoint(
        2, {x + cos_a * rx - sin_a * ry, y + sin_a * rx + cos_a * ry});

    triangle_.setFillColor(base_color);
    triangle_.setOutlineColor(outline_color);
    triangle_.setOutlineThickness(selected ? visual::SELECTED_OUTLINE_THICKNESS
                                           : 0.0f);

    target.draw(triangle_);
  } else {
    float radius = visual_radius;
    circle_.setRadius(radius);
    circle_.setOrigin({radius, radius});
    circle_.setPosition({agent.position.x, agent.position.y});
    circle_.setPointCount(visual::CIRCLE_POINT_COUNT);

    circle_.setFillColor(base_color);
    circle_.setOutlineColor(outline_color);
    circle_.setOutlineThickness(selected ? visual::SELECTED_OUTLINE_THICKNESS
                                         : 0.0f);

    target.draw(circle_);
  }
}

void Renderer::draw_all_agents(sf::RenderTarget &target,
                               const std::vector<RenderAgent> &agents,
                               Entity selected_entity) {
  for (const auto &agent : agents) {
    draw_agent(target, agent, agent.entity == selected_entity);
  }
}

void Renderer::draw_vision_range(sf::RenderTarget &target, Vec2 position,
                                 float vision_range) {
  sf::CircleShape vision(vision_range, visual::VISION_POINT_COUNT);
  vision.setOrigin({vision_range, vision_range});
  vision.setPosition({position.x, position.y});
  vision.setFillColor(sf::Color(visual::VISION_FILL_R, visual::VISION_FILL_G,
                                visual::VISION_FILL_B,
                                visual::VISION_FILL_ALPHA));
  vision.setOutlineColor(sf::Color(visual::VISION_FILL_R, visual::VISION_FILL_G,
                                   visual::VISION_FILL_B,
                                   visual::VISION_OUTLINE_ALPHA));
  vision.setOutlineThickness(1.0f);
  target.draw(vision);
}

void Renderer::draw_sensor_lines(sf::RenderTarget &target,
                                 const std::vector<RenderLine> &lines_in) {
  sf::VertexArray lines(sf::PrimitiveType::Lines);
  for (const auto &line : lines_in) {
    lines.append(sf::Vertex{{line.start.x, line.start.y}, line.color});
    lines.append(sf::Vertex{{line.end.x, line.end.y}, line.color});
  }

  target.draw(lines);
}

} // namespace moonai
