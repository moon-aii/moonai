#include "visualization/visualization_manager.hpp"

#include "core/profiler_macros.hpp"

#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <SFML/Window/WindowEnums.hpp>

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace moonai {

VisualizationManager::VisualizationManager(const SimulationConfig &config, UiState &ui_state)
    : config_(config), ui_state_(ui_state) {}

VisualizationManager::~VisualizationManager() = default;

bool VisualizationManager::initialize() {
  window_ = std::make_unique<sf::RenderWindow>(sf::VideoMode({window_width_, window_height_}), "MoonAI");
  window_->setFramerateLimit(VisualizationManager::kGuiMaxFps);

  // Set up camera to show the simulation world
  camera_view_ =
      sf::View(sf::Vector2f(static_cast<float>(config_.grid_size) / 2.0f, static_cast<float>(config_.grid_size) / 2.0f),
               sf::Vector2f(static_cast<float>(config_.grid_size), static_cast<float>(config_.grid_size)));

  // Adjust view to maintain aspect ratio with margin around simulation
  float margin = VisualizationManager::simulation_margin();
  float world_aspect = static_cast<float>(config_.grid_size) / static_cast<float>(config_.grid_size);
  float window_aspect = static_cast<float>(window_width_) / window_height_;

  if (window_aspect > world_aspect) {
    camera_view_.setSize(sf::Vector2f(static_cast<float>(config_.grid_size) * window_aspect + 2.0f * margin,
                                      static_cast<float>(config_.grid_size) + 2.0f * margin));
  } else {
    camera_view_.setSize(sf::Vector2f(static_cast<float>(config_.grid_size) + 2.0f * margin,
                                      static_cast<float>(config_.grid_size) / window_aspect + 2.0f * margin));
  }

  window_->setView(camera_view_);

  overlay_.initialize();
  frame_.overlay_stats.max_steps = config_.max_steps;

  running_ = true;
  spdlog::info("Visualization initialized ({}x{} window, {}x{} world)", window_width_, window_height_,
               config_.grid_size, config_.grid_size);
  return true;
}

void VisualizationManager::render(FrameSnapshot frame) {
  MOONAI_PROFILE_SCOPE("render");

  if (!window_ || !running_)
    return;

  frame_ = std::move(frame);

  // Apply any pending click now that we have registry access
  if (pending_click_) {
    handle_mouse_click(pending_click_x_, pending_click_y_);
    pending_click_ = false;
  }

  window_->clear(sf::Color::Black);
  window_->setView(camera_view_);

  {
    MOONAI_PROFILE_SCOPE("render_world");
    renderer_.draw_background(*window_, frame_.world_width, frame_.world_height);
    renderer_.draw_grid(*window_, frame_.world_width, frame_.world_height, 300.0f);
    renderer_.draw_boundaries(*window_, frame_.world_width, frame_.world_height);
    renderer_.draw_food(*window_, frame_.foods);
  }

  {
    MOONAI_PROFILE_SCOPE("render_agents");
    renderer_.draw_predators(*window_, frame_.predators, ui_state_.selected_agent_id);
    renderer_.draw_prey(*window_, frame_.prey, ui_state_.selected_agent_id);
  }

  if (frame_.has_selected_vision && frame_.selected_agent_id == ui_state_.selected_agent_id) {
    MOONAI_PROFILE_SCOPE("render_sensor_lines");
    Renderer::draw_vision_range(*window_, frame_.selected_position, frame_.selected_vision_range);
    Renderer::draw_sensor_lines(*window_, frame_.sensor_lines);
  }

  update_fps(frame_clock_.restart().asSeconds());
  frame_.overlay_stats.fps = current_fps_;

  if (frame_.overlay_stats.step != last_chart_step_) {
    overlay_.push_population(frame_.overlay_stats.alive_predator, frame_.overlay_stats.alive_prey,
                             frame_.overlay_stats.active_food);
    overlay_.push_complexity(frame_.overlay_stats.avg_predator_complexity, frame_.overlay_stats.avg_prey_complexity);
    overlay_.push_energy(frame_.overlay_stats.avg_predator_energy, frame_.overlay_stats.avg_prey_energy);
    last_chart_step_ = frame_.overlay_stats.step;
  }

  {
    MOONAI_PROFILE_SCOPE("render_ui");
    overlay_.set_activations(frame_.selected_node_activations);
    overlay_.draw(*window_, frame_.overlay_stats, frame_.selected_genome);
  }

  {
    MOONAI_PROFILE_SCOPE("swap_buffers");
    window_->display();
  }
}

bool VisualizationManager::should_close() const {
  return !running_;
}

void VisualizationManager::handle_events() {
  MOONAI_PROFILE_SCOPE("handle_events");

  if (!window_)
    return;

  while (const auto event = window_->pollEvent()) {
    // Window close
    if (event->is<sf::Event::Closed>()) {
      running_ = false;
      return;
    }

    // Key pressed
    if (const auto *key = event->getIf<sf::Event::KeyPressed>()) {
      switch (key->code) {
        case sf::Keyboard::Key::Escape:
          running_ = false;
          break;

        case sf::Keyboard::Key::Space:
          ui_state_.paused = !ui_state_.paused;
          break;

        case sf::Keyboard::Key::Period: // > key (step forward)
          if (ui_state_.paused)
            ui_state_.step_requested = true;
          break;

        case sf::Keyboard::Key::Equal: // + key
        case sf::Keyboard::Key::Up:
        case sf::Keyboard::Key::Add:
          ui_state_.speed_multiplier = std::min(ui_state_.speed_multiplier * 2, 1024);
          break;

        case sf::Keyboard::Key::Hyphen: // - key
        case sf::Keyboard::Key::Down:
        case sf::Keyboard::Key::Subtract:
          ui_state_.speed_multiplier = std::max(ui_state_.speed_multiplier / 2, 1);
          break;

        case sf::Keyboard::Key::S:
          if (window_) {
            sf::Texture texture(window_->getSize());
            texture.update(*window_);
            sf::Image img = texture.copyToImage();
            std::string fname = "screenshot_step" + std::to_string(frame_.overlay_stats.step) + ".png";
            (void)img.saveToFile(fname);
            spdlog::info("Screenshot saved: {}", fname);
          }
          break;

        case sf::Keyboard::Key::Home:
          // Reset camera to default
          zoom_level_ = 1.0f;
          camera_view_.setCenter(
              sf::Vector2f(static_cast<float>(config_.grid_size) / 2.0f, static_cast<float>(config_.grid_size) / 2.0f));
          update_camera();
          break;

        default:
          break;
      }
    }

    if (const auto *scroll = event->getIf<sf::Event::MouseWheelScrolled>()) {
      float delta = scroll->delta;
      float factor = (delta > 0) ? 0.9f : 1.1f;
      zoom_level_ *= factor;
      zoom_level_ = std::clamp(zoom_level_, 0.1f, 10.0f);
      update_camera();
    }

    if (const auto *btn = event->getIf<sf::Event::MouseButtonPressed>()) {
      if (btn->button == sf::Mouse::Button::Middle || btn->button == sf::Mouse::Button::Right) {
        dragging_ = true;
        drag_start_ = sf::Vector2f(static_cast<float>(btn->position.x), static_cast<float>(btn->position.y));
        view_start_ = camera_view_.getCenter();
      }
    }

    if (const auto *btn = event->getIf<sf::Event::MouseButtonReleased>()) {
      if (btn->button == sf::Mouse::Button::Middle || btn->button == sf::Mouse::Button::Right) {
        dragging_ = false;
      }

      if (btn->button == sf::Mouse::Button::Left && window_) {
        if (btn->position.x >= static_cast<int>(ui_side_margin() + 10.0f)) {
          auto world_pos = window_->mapPixelToCoords(btn->position, camera_view_);
          pending_click_ = true;
          pending_click_x_ = world_pos.x;
          pending_click_y_ = world_pos.y;
        }
      }
    }

    if (const auto *moved = event->getIf<sf::Event::MouseMoved>()) {
      if (dragging_ && window_) {
        sf::Vector2f current(static_cast<float>(moved->position.x), static_cast<float>(moved->position.y));
        sf::Vector2f delta = drag_start_ - current;
        // Scale by zoom level
        delta.x *= zoom_level_;
        delta.y *= zoom_level_;
        camera_view_.setCenter(view_start_ + delta);
      }
    }

    if (const auto *resized = event->getIf<sf::Event::Resized>()) {
      window_width_ = resized->size.x;
      window_height_ = resized->size.y;
      update_camera();
    }
  }
}

void VisualizationManager::handle_mouse_click(float world_x, float world_y) {
  float best_dist = 60.0f * zoom_level_;
  uint32_t best_agent_id = 0;

  for (const auto &agent : frame_.predators) {
    float dx = agent.position.x - world_x;
    float dy = agent.position.y - world_y;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist < best_dist) {
      best_dist = dist;
      best_agent_id = agent.agent_id;
    }
  }

  for (const auto &agent : frame_.prey) {
    float dx = agent.position.x - world_x;
    float dy = agent.position.y - world_y;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist < best_dist) {
      best_dist = dist;
      best_agent_id = agent.agent_id;
    }
  }

  ui_state_.selected_agent_id = best_agent_id;
}

void VisualizationManager::update_camera() {
  if (!window_)
    return;

  float world_w = static_cast<float>(config_.grid_size);
  float world_h = static_cast<float>(config_.grid_size);

  float window_aspect = static_cast<float>(window_width_) / window_height_;
  float world_aspect = world_w / world_h;

  float view_w, view_h;
  float margin = VisualizationManager::simulation_margin();
  if (window_aspect > world_aspect) {
    view_h = world_h * zoom_level_;
    view_w = view_h * window_aspect;
  } else {
    view_w = world_w * zoom_level_;
    view_h = view_w / window_aspect;
  }

  // Add margin around simulation area
  camera_view_.setSize(sf::Vector2f(view_w + 2.0f * margin, view_h + 2.0f * margin));

  // Camera stays centered - left and right UI panels are equal width (300px
  // each)
}

void VisualizationManager::update_fps(float dt) {
  if (dt > 0.0f) {
    float instant_fps = 1.0f / dt;
    current_fps_ = current_fps_ * (1.0f - fps_alpha_) + instant_fps * fps_alpha_;
  }
}

} // namespace moonai
