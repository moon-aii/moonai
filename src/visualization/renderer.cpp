#include "visualization/renderer.hpp"

#include <SFML/Graphics/Vertex.hpp>
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>

#include <cmath>

namespace moonai {

Renderer::Renderer() {
    // Pre-configure triangle shape (3 points for predator)
    triangle_.setPointCount(3);

    // Pre-configure circle for prey
    circle_.setPointCount(20);
}

void Renderer::draw_background(sf::RenderTarget& target, int width, int height) {
    rect_.setSize({static_cast<float>(width), static_cast<float>(height)});
    rect_.setPosition({0.0f, 0.0f});
    rect_.setFillColor(sf::Color(20, 20, 30));
    target.draw(rect_);
}

void Renderer::draw_grid(sf::RenderTarget& target, int width, int height, float cell_size) {
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

void Renderer::draw_boundaries(sf::RenderTarget& target, int width, int height) {
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

void Renderer::draw_food(sf::RenderTarget& target, const std::vector<Food>& food) {
    circle_.setRadius(3.0f);
    circle_.setOrigin({3.0f, 3.0f});

    for (const auto& f : food) {
        if (!f.active) continue;
        circle_.setPosition({f.position.x, f.position.y});
        circle_.setFillColor(sf::Color(100, 200, 50, 180));
        circle_.setOutlineThickness(0);
        target.draw(circle_);
    }
}

void Renderer::draw_agent(sf::RenderTarget& target, const Agent& agent, bool selected) {
    float alpha = agent.alive() ? 255.0f : dead_fade_alpha;
    auto a = static_cast<std::uint8_t>(alpha);

    // Use species color when available, otherwise fall back to type color
    sf::Color base_color;
    if (agent.species_id() >= 0) {
        base_color = species_color(agent.species_id());
    } else if (agent.type() == AgentType::Predator) {
        base_color = sf::Color(220, 60, 60);
    } else {
        base_color = sf::Color(60, 200, 80);
    }
    base_color.a = a;

    if (selected) {
        // Brighten for selection
        base_color.r = std::min(255, base_color.r + 60);
        base_color.g = std::min(255, base_color.g + 60);
        base_color.b = std::min(255, base_color.b + 60);
    }

    sf::Color outline_color(
        std::min(255, base_color.r + 30),
        std::min(255, base_color.g + 30),
        std::min(255, base_color.b + 30), a);

    if (agent.type() == AgentType::Predator) {
        // Draw predator as triangle pointing in movement direction
        float size = 8.0f;
        Vec2 vel = agent.velocity();
        float angle = std::atan2(vel.y, vel.x);

        float x = agent.position().x;
        float y = agent.position().y;

        // Triangle vertices relative to center, rotated by angle
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        // Tip (forward)
        triangle_.setPoint(0, {x + cos_a * size * 1.5f, y + sin_a * size * 1.5f});
        // Left rear
        float lx = -size * 0.8f, ly = -size * 0.7f;
        triangle_.setPoint(1, {x + cos_a * lx - sin_a * ly,
                                y + sin_a * lx + cos_a * ly});
        // Right rear
        float rx = -size * 0.8f, ry = size * 0.7f;
        triangle_.setPoint(2, {x + cos_a * rx - sin_a * ry,
                                y + sin_a * rx + cos_a * ry});

        triangle_.setFillColor(base_color);
        triangle_.setOutlineColor(outline_color);
        triangle_.setOutlineThickness(selected ? 2.0f : 0.0f);

        target.draw(triangle_);
    } else {
        // Draw prey as circle
        float radius = 5.0f;
        circle_.setRadius(radius);
        circle_.setOrigin({radius, radius});
        circle_.setPosition({agent.position().x, agent.position().y});
        circle_.setPointCount(20);

        circle_.setFillColor(base_color);
        circle_.setOutlineColor(outline_color);
        circle_.setOutlineThickness(selected ? 2.0f : 0.0f);

        target.draw(circle_);
    }
}

void Renderer::draw_vision_range(sf::RenderTarget& target, const Agent& agent) {
    float r = agent.vision_range();
    sf::CircleShape vision(r, 60);
    vision.setOrigin({r, r});
    vision.setPosition({agent.position().x, agent.position().y});
    vision.setFillColor(sf::Color(255, 255, 255, 15));
    vision.setOutlineColor(sf::Color(255, 255, 255, 40));
    vision.setOutlineThickness(1.0f);
    target.draw(vision);
}

void Renderer::draw_sensor_lines(sf::RenderTarget& target, const Agent& agent,
                                  const std::vector<std::unique_ptr<Agent>>& agents,
                                  const std::vector<Food>& food) {
    Vec2 pos = agent.position();
    float vision = agent.vision_range();

    sf::VertexArray lines(sf::PrimitiveType::Lines);

    // Lines to nearby agents
    for (const auto& other : agents) {
        if (!other->alive() || other->id() == agent.id()) continue;
        Vec2 diff = other->position() - pos;
        if (diff.length() > vision) continue;

        sf::Color line_color;
        if (other->type() == AgentType::Predator) {
            line_color = sf::Color(255, 80, 80, 80);
        } else {
            line_color = sf::Color(80, 255, 80, 80);
        }

        lines.append(sf::Vertex{{pos.x, pos.y}, line_color});
        lines.append(sf::Vertex{{other->position().x, other->position().y}, line_color});
    }

    // Lines to nearby food
    for (const auto& f : food) {
        if (!f.active) continue;
        Vec2 diff = f.position - pos;
        if (diff.length() > vision) continue;

        sf::Color food_line(200, 200, 50, 60);
        lines.append(sf::Vertex{{pos.x, pos.y}, food_line});
        lines.append(sf::Vertex{{f.position.x, f.position.y}, food_line});
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
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }

    return sf::Color(static_cast<std::uint8_t>(r * 255),
                     static_cast<std::uint8_t>(g * 255),
                     static_cast<std::uint8_t>(b * 255));
}

} // namespace moonai
