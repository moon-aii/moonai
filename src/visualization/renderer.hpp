#pragma once

#include "core/types.hpp"

#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/Vertex.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace moonai {

struct RenderFood {
  Vec2 position;
};

struct RenderAgent {
  uint32_t entity = INVALID_ENTITY;
  uint32_t agent_id = 0;
  Vec2 position;
  Vec2 velocity;
};

struct RenderLine {
  Vec2 start;
  Vec2 end;
  sf::Color color;
};

class Renderer {
public:
  Renderer();

  void draw_background(sf::RenderTarget &target, int width, int height);
  static void draw_grid(sf::RenderTarget &target, int width, int height, float cell_size);
  static void draw_boundaries(sf::RenderTarget &target, int width, int height);

  void draw_food(sf::RenderTarget &target, const std::vector<RenderFood> &food);
  void draw_predators(sf::RenderTarget &target, const std::vector<RenderAgent> &predators,
                      uint32_t selected_agent_id = 0);
  void draw_prey(sf::RenderTarget &target, const std::vector<RenderAgent> &prey, uint32_t selected_agent_id = 0);
  static void draw_vision_range(sf::RenderTarget &target, Vec2 position, float vision_range);
  static void draw_sensor_lines(sf::RenderTarget &target, const std::vector<RenderLine> &lines);

private:
  static constexpr int kPreyCircleSegments = 6;

  void initialize_circle_template();
  static sf::Color brighten_color(sf::Color color, int amount);
  void draw_triangles(sf::RenderTarget &target, const std::vector<sf::Vertex> &vertices) const;
  static void write_quad(sf::Vertex *vertices, Vec2 position, float half_extent, sf::Color color);
  static void write_triangle(sf::Vertex *vertices, const RenderAgent &agent, float size, sf::Color color);
  void write_circle(sf::Vertex *vertices, Vec2 position, float radius, sf::Color color) const;

  std::vector<sf::Vertex> food_vertices_;
  std::vector<sf::Vertex> predator_vertices_;
  std::vector<sf::Vertex> prey_vertices_;
  std::vector<sf::Vertex> selected_vertices_;
  sf::RectangleShape rect_;
  std::array<Vec2, kPreyCircleSegments + 1> prey_circle_template_{};
};

} // namespace moonai
