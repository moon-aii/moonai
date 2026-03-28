#pragma once

#include "core/types.hpp"
#include "simulation/entity.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/ConvexShape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderTarget.hpp>

#include <cstdint>
#include <vector>

namespace moonai {

struct RenderFood {
  Vec2 position;
};

struct RenderAgent {
  Entity entity = INVALID_ENTITY;
  Vec2 position;
  Vec2 velocity;
  uint8_t type = 0;
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
  static void draw_grid(sf::RenderTarget &target, int width, int height,
                        float cell_size);
  static void draw_boundaries(sf::RenderTarget &target, int width, int height);

  void draw_food(sf::RenderTarget &target, const std::vector<RenderFood> &food);
  void draw_agent(sf::RenderTarget &target, const RenderAgent &agent,
                  bool selected = false);
  void draw_all_agents(sf::RenderTarget &target,
                       const std::vector<RenderAgent> &agents,
                       Entity selected_entity = INVALID_ENTITY);
  static void draw_vision_range(sf::RenderTarget &target, Vec2 position,
                                float vision_range);
  static void draw_sensor_lines(sf::RenderTarget &target,
                                const std::vector<RenderLine> &lines);

private:
  sf::CircleShape circle_;
  sf::ConvexShape triangle_;
  sf::RectangleShape rect_;
};

} // namespace moonai
