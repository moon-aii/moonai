#pragma once

#include "core/types.hpp"
#include "simulation/entity.hpp"
#include "simulation/registry.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/ConvexShape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderTarget.hpp>

#include <cstdint>

namespace moonai {

class Renderer {
public:
  Renderer();

  void draw_background(sf::RenderTarget &target, int width, int height);
  static void draw_grid(sf::RenderTarget &target, int width, int height,
                        float cell_size);
  static void draw_boundaries(sf::RenderTarget &target, int width, int height);

  void draw_food_ecs(sf::RenderTarget &target, const Registry &registry);
  void draw_agent_ecs(sf::RenderTarget &target, const Registry &registry,
                      Entity entity, bool selected = false);
  void draw_all_agents_ecs(sf::RenderTarget &target, const Registry &registry,
                           Entity selected_entity = INVALID_ENTITY);
  static void draw_vision_range_ecs(sf::RenderTarget &target,
                                    const Registry &registry, Entity entity);
  static void draw_sensor_lines_ecs(sf::RenderTarget &target,
                                    const Registry &registry, Entity entity);

  static sf::Color species_color(int species_id);

  float dead_fade_alpha = 60.0f;

private:
  sf::CircleShape circle_;
  sf::ConvexShape triangle_;
  sf::RectangleShape rect_;
};

} // namespace moonai
