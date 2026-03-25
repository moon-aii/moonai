#include "visualization/renderer.hpp"
#include "simulation/registry.hpp"

#include <SFML/Graphics/PrimitiveType.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <cmath>

namespace moonai {

namespace ecs {
class Registry;
}

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
  rect_.setFillColor(sf::Color(20, 20, 30));
  target.draw(rect_);
}

void Renderer::draw_grid(sf::RenderTarget &target, int width, int height,
                         float cell_size) {
  sf::VertexArray lines(sf::PrimitiveType::Lines);
  sf::Color grid_color(40, 40, 55);

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
  sf::Color border_color(100, 100, 140);

  float w = static_cast<float>(width);
  float h = static_cast<float>(height);
  border[0] = sf::Vertex{{0, 0}, border_color};
  border[1] = sf::Vertex{{w, 0}, border_color};
  border[2] = sf::Vertex{{w, h}, border_color};
  border[3] = sf::Vertex{{0, h}, border_color};
  border[4] = sf::Vertex{{0, 0}, border_color};

  target.draw(border);
}

void Renderer::draw_food_ecs(sf::RenderTarget &target,
                             const Registry &registry) {
  circle_.setRadius(3.0f);
  circle_.setOrigin({3.0f, 3.0f});

  const auto &living = registry.living_entities();
  const auto &positions = registry.positions();
  const auto &identity = registry.identity();
  const auto &food_state = registry.food_state();
  const auto &visual = registry.visual();

  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);

    // Only draw food entities
    if (identity.type[idx] != IdentitySoA::TYPE_FOOD) {
      continue;
    }

    // Only draw active food
    if (!food_state.active[idx]) {
      continue;
    }

    // Use visual color if available, otherwise use default green
    sf::Color color(100, 200, 50, 180);
    if (idx < visual.color_rgba.size() && visual.color_rgba[idx] != 0) {
      uint32_t rgba = visual.color_rgba[idx];
      color.r = static_cast<uint8_t>((rgba >> 24) & 0xFF);
      color.g = static_cast<uint8_t>((rgba >> 16) & 0xFF);
      color.b = static_cast<uint8_t>((rgba >> 8) & 0xFF);
      color.a = static_cast<uint8_t>(rgba & 0xFF);
    }

    circle_.setPosition({positions.x[idx], positions.y[idx]});
    circle_.setFillColor(color);
    circle_.setOutlineThickness(0);
    target.draw(circle_);
  }
}

void Renderer::draw_agent_ecs(sf::RenderTarget &target,
                              const Registry &registry, Entity entity,
                              bool selected) {
  if (!registry.valid(entity)) {
    return;
  }

  size_t idx = registry.index_of(entity);
  const auto &vitals = registry.vitals();
  const auto &positions = registry.positions();
  const auto &motion = registry.motion();
  const auto &identity = registry.identity();
  const auto &visual = registry.visual();

  bool alive = vitals.alive[idx] != 0;
  float alpha = alive ? 255.0f : dead_fade_alpha;
  auto a = static_cast<std::uint8_t>(alpha);

  // Use VisualSoA colors if available, otherwise fall back to type-based colors
  sf::Color base_color;
  if (idx < visual.color_rgba.size() && visual.color_rgba[idx] != 0) {
    // Unpack RGBA from uint32_t
    uint32_t rgba = visual.color_rgba[idx];
    base_color.r = static_cast<uint8_t>((rgba >> 24) & 0xFF);
    base_color.g = static_cast<uint8_t>((rgba >> 16) & 0xFF);
    base_color.b = static_cast<uint8_t>((rgba >> 8) & 0xFF);
    base_color.a = static_cast<uint8_t>(rgba & 0xFF);
  } else {
    // Fallback to type-based colors
    if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
      base_color = sf::Color(220, 60, 60);
    } else {
      base_color = sf::Color(60, 200, 80);
    }
  }
  base_color.a = static_cast<uint8_t>(base_color.a * alpha / 255.0f);

  if (selected) {
    // Brighten for selection
    base_color.r = std::min(255, base_color.r + 60);
    base_color.g = std::min(255, base_color.g + 60);
    base_color.b = std::min(255, base_color.b + 60);
  }

  sf::Color outline_color(std::min(255, base_color.r + 30),
                          std::min(255, base_color.g + 30),
                          std::min(255, base_color.b + 30), a);

  // Use visual radius if available, otherwise use defaults
  float visual_radius = (idx < visual.radius.size() && visual.radius[idx] > 0)
                            ? visual.radius[idx]
                            : 8.0f;

  // Use shape_type if available, otherwise default based on agent type
  // shape == 0 means "use default", shape == 1 means triangle, shape == 2 means
  // circle
  uint8_t shape = (idx < visual.shape_type.size()) ? visual.shape_type[idx] : 0;
  bool is_triangle =
      (shape == 1) ||
      (shape == 0 && identity.type[idx] == IdentitySoA::TYPE_PREDATOR);

  if (is_triangle) {
    // Draw as triangle pointing in movement direction
    float size = visual_radius;
    Vec2 vel{motion.vel_x[idx], motion.vel_y[idx]};
    float angle = std::atan2(vel.y, vel.x);

    float x = positions.x[idx];
    float y = positions.y[idx];

    // Triangle vertices relative to center, rotated by angle
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);

    // Tip (forward)
    triangle_.setPoint(0, {x + cos_a * size * 1.5f, y + sin_a * size * 1.5f});
    // Left rear
    float lx = -size * 0.8f, ly = -size * 0.7f;
    triangle_.setPoint(
        1, {x + cos_a * lx - sin_a * ly, y + sin_a * lx + cos_a * ly});
    // Right rear
    float rx = -size * 0.8f, ry = size * 0.7f;
    triangle_.setPoint(
        2, {x + cos_a * rx - sin_a * ry, y + sin_a * rx + cos_a * ry});

    triangle_.setFillColor(base_color);
    triangle_.setOutlineColor(outline_color);
    triangle_.setOutlineThickness(selected ? 2.0f : 0.0f);

    target.draw(triangle_);
  } else {
    // Draw as circle
    float radius = visual_radius;
    circle_.setRadius(radius);
    circle_.setOrigin({radius, radius});
    circle_.setPosition({positions.x[idx], positions.y[idx]});
    circle_.setPointCount(20);

    circle_.setFillColor(base_color);
    circle_.setOutlineColor(outline_color);
    circle_.setOutlineThickness(selected ? 2.0f : 0.0f);

    target.draw(circle_);
  }
}

void Renderer::draw_all_agents_ecs(sf::RenderTarget &target,
                                   const Registry &registry,
                                   Entity selected_entity) {
  const auto &living = registry.living_entities();
  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    const auto &vitals = registry.vitals();
    if (vitals.alive[idx]) {
      bool is_selected = (entity == selected_entity);
      draw_agent_ecs(target, registry, entity, is_selected);
    }
  }

  // Draw dead agents (faded)
  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    const auto &vitals = registry.vitals();
    if (!vitals.alive[idx]) {
      bool is_selected = (entity == selected_entity);
      draw_agent_ecs(target, registry, entity, is_selected);
    }
  }
}

void Renderer::draw_vision_range_ecs(sf::RenderTarget &target,
                                     const Registry &registry, Entity entity) {
  if (!registry.valid(entity)) {
    return;
  }

  size_t idx = registry.index_of(entity);
  const auto &positions = registry.positions();
  const auto &sensors = registry.sensors();

  // Get sensor range (use a default or from config)
  float r = 100.0f; // Default vision range
  if (idx * SensorSoA::INPUT_COUNT < sensors.inputs.size()) {
    // Could extract from sensor config if stored there
    // For now use default
  }

  sf::CircleShape vision(r, 60);
  vision.setOrigin({r, r});
  vision.setPosition({positions.x[idx], positions.y[idx]});
  vision.setFillColor(sf::Color(255, 255, 255, 15));
  vision.setOutlineColor(sf::Color(255, 255, 255, 40));
  vision.setOutlineThickness(1.0f);
  target.draw(vision);
}

void Renderer::draw_sensor_lines_ecs(sf::RenderTarget &target,
                                     const Registry &registry, Entity entity) {
  if (!registry.valid(entity)) {
    return;
  }

  size_t idx = registry.index_of(entity);
  const auto &positions = registry.positions();
  const auto &vitals = registry.vitals();
  const auto &identity = registry.identity();
  const auto &sensors = registry.sensors();

  if (!vitals.alive[idx]) {
    return;
  }

  Vec2 pos{positions.x[idx], positions.y[idx]};
  float vision = 100.0f; // Default vision range

  sf::VertexArray lines(sf::PrimitiveType::Lines);

  // We need a spatial query to find nearby entities
  // For now, iterate all entities (optimization: use spatial grid)
  const auto &living = registry.living_entities();
  for (Entity other_entity : living) {
    if (other_entity == entity) {
      continue;
    }

    size_t other_idx = registry.index_of(other_entity);
    const auto &other_vitals = registry.vitals();

    if (!other_vitals.alive[other_idx]) {
      continue;
    }

    const auto &other_positions = registry.positions();
    Vec2 other_pos{other_positions.x[other_idx], other_positions.y[other_idx]};

    Vec2 diff = other_pos - pos;
    if (diff.length() > vision) {
      continue;
    }

    const auto &other_identity = registry.identity();
    sf::Color line_color;
    if (other_identity.type[other_idx] == IdentitySoA::TYPE_PREDATOR) {
      line_color = sf::Color(255, 80, 80, 80);
    } else {
      line_color = sf::Color(80, 255, 80, 80);
    }

    lines.append(sf::Vertex{{pos.x, pos.y}, line_color});
    lines.append(sf::Vertex{{other_pos.x, other_pos.y}, line_color});
  }

  // Lines to nearby food
  const auto &food_state = registry.food_state();
  for (Entity food_entity : living) {
    size_t food_idx = registry.index_of(food_entity);
    if (identity.type[food_idx] != IdentitySoA::TYPE_FOOD) {
      continue;
    }
    if (!food_state.active[food_idx]) {
      continue;
    }
    const auto &food_positions = registry.positions();
    Vec2 food_pos{food_positions.x[food_idx], food_positions.y[food_idx]};
    Vec2 diff = food_pos - pos;
    if (diff.length() > vision)
      continue;

    sf::Color food_line(200, 200, 50, 60);
    lines.append(sf::Vertex{{pos.x, pos.y}, food_line});
    lines.append(sf::Vertex{{food_pos.x, food_pos.y}, food_line});
  }

  target.draw(lines);
}

sf::Color Renderer::species_color(int species_id) {
  // Generate distinct colors from species ID using golden ratio hue spacing
  float hue = std::fmod(species_id * 137.508f, 360.0f);
  float s = 0.7f, v = 0.9f;

  // HSV to RGB
  int hi = static_cast<int>(hue / 60.0f) % 6;
  float f = hue / 60.0f - static_cast<float>(hi);
  float p = v * (1.0f - s);
  float q = v * (1.0f - f * s);
  float t = v * (1.0f - (1.0f - f) * s);

  float r, g, b;
  switch (hi) {
    case 0:
      r = v;
      g = t;
      b = p;
      break;
    case 1:
      r = q;
      g = v;
      b = p;
      break;
    case 2:
      r = p;
      g = v;
      b = t;
      break;
    case 3:
      r = p;
      g = q;
      b = v;
      break;
    case 4:
      r = t;
      g = p;
      b = v;
      break;
    default:
      r = v;
      g = p;
      b = q;
      break;
  }

  return sf::Color(static_cast<std::uint8_t>(r * 255),
                   static_cast<std::uint8_t>(g * 255),
                   static_cast<std::uint8_t>(b * 255));
}

} // namespace moonai
