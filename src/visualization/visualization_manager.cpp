#include "visualization/visualization_manager.hpp"
#include "visualization/constants.hpp"

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

VisualizationManager::VisualizationManager(const SimulationConfig &config,
                                           UiState &ui_state)
    : config_(config), ui_state_(ui_state) {}

VisualizationManager::~VisualizationManager() = default;

bool VisualizationManager::initialize() {
  window_ = std::make_unique<sf::RenderWindow>(
      sf::VideoMode({window_width_, window_height_}),
      "MoonAI - Predator-Prey Evolution");
  window_->setFramerateLimit(VisualizationManager::kGuiMaxFps);

  // Set up camera to show the simulation world
  camera_view_ =
      sf::View(sf::Vector2f(static_cast<float>(config_.grid_size) / 2.0f,
                            static_cast<float>(config_.grid_size) / 2.0f),
               sf::Vector2f(static_cast<float>(config_.grid_size),
                            static_cast<float>(config_.grid_size)));

  // Adjust view to maintain aspect ratio with margin around simulation
  float margin = VisualizationManager::simulation_margin();
  float world_aspect = static_cast<float>(config_.grid_size) /
                       static_cast<float>(config_.grid_size);
  float window_aspect = static_cast<float>(window_width_) / window_height_;

  if (window_aspect > world_aspect) {
    camera_view_.setSize(sf::Vector2f(
        static_cast<float>(config_.grid_size) * window_aspect + 2.0f * margin,
        static_cast<float>(config_.grid_size) + 2.0f * margin));
  } else {
    camera_view_.setSize(sf::Vector2f(
        static_cast<float>(config_.grid_size) + 2.0f * margin,
        static_cast<float>(config_.grid_size) / window_aspect + 2.0f * margin));
  }

  window_->setView(camera_view_);

  overlay_.initialize();
  frame_.overlay_stats.max_steps = config_.max_steps;

  running_ = true;
  spdlog::info("Visualization initialized ({}x{} window, {}x{} world)",
               window_width_, window_height_, config_.grid_size,
               config_.grid_size);
  return true;
}

void VisualizationManager::set_experiments(
    const std::vector<std::string> &names) {
  experiment_names_ = names;
  if (!names.empty()) {
    experiment_select_mode_ = true;
    experiment_scroll_offset_ = 0;
    experiment_hover_index_ = -1;
  }
}

void VisualizationManager::enter_experiment_select_mode() {
  if (!experiment_names_.empty()) {
    experiment_select_mode_ = true;
    experiment_scroll_offset_ = 0;
    experiment_hover_index_ = -1;
    ui_state_.paused = true;
  }
}

void VisualizationManager::render(FrameSnapshot frame) {
  if (!window_ || !running_)
    return;

  frame_ = std::move(frame);

  // Experiment selector mode: render selector overlay only
  if (experiment_select_mode_) {
    window_->clear(sf::Color(visual::BG_R, visual::BG_G, visual::BG_B));
    overlay_.draw_experiment_selector(*window_, experiment_names_,
                                      experiment_hover_index_,
                                      experiment_scroll_offset_);
    window_->display();
    return;
  }

  // Apply any pending click now that we have registry access
  if (pending_click_) {
    handle_mouse_click(pending_click_x_, pending_click_y_);
    pending_click_ = false;
  }

  window_->clear(sf::Color::Black);
  window_->setView(camera_view_);

  // Draw world
  {
    MOONAI_PROFILE_SCOPE("render_world");
    renderer_.draw_background(*window_, frame_.world_width,
                              frame_.world_height);
    renderer_.draw_grid(*window_, frame_.world_width, frame_.world_height,
                        500.0f);
    renderer_.draw_boundaries(*window_, frame_.world_width,
                              frame_.world_height);
    renderer_.draw_food(*window_, frame_.foods);
  }

  // Draw all agents using ECS
  {
    MOONAI_PROFILE_SCOPE("render_agents");
    renderer_.draw_all_agents(
        *window_, frame_.agents, frame_.overlay_stats.alive_predators,
        frame_.overlay_stats.alive_prey, ui_state_.selected_agent_id);
  }

  // Draw vision/sensor lines for selected entity (automatically shown when
  // agent is clicked)
  if (frame_.has_selected_vision &&
      frame_.selected_agent_id == ui_state_.selected_agent_id) {
    MOONAI_PROFILE_SCOPE("render_sensor_lines");
    Renderer::draw_vision_range(*window_, frame_.selected_position,
                                frame_.selected_vision_range);
    Renderer::draw_sensor_lines(*window_, frame_.sensor_lines);
  }

  // Update FPS counter using exponential moving average
  update_fps(frame_clock_.restart().asSeconds());
  frame_.overlay_stats.fps = current_fps_;

  if (ui_state_.selected_agent_id != 0) {
    frame_.overlay_stats.selected_agent =
        static_cast<int>(ui_state_.selected_agent_id);
  }

  if (frame_.overlay_stats.step != last_chart_step_) {
    overlay_.push_population(frame_.overlay_stats.alive_predators,
                             frame_.overlay_stats.alive_prey,
                             frame_.overlay_stats.active_food);
    last_chart_step_ = frame_.overlay_stats.step;
  }

  // Draw UI overlay (with selected genome for NN topology panel)
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
  if (!window_)
    return;

  while (const auto event = window_->pollEvent()) {
    // Window close
    if (event->is<sf::Event::Closed>()) {
      running_ = false;
      return;
    }

    // Experiment selector mode input handling
    if (experiment_select_mode_) {
      if (const auto *key = event->getIf<sf::Event::KeyPressed>()) {
        if (key->code == sf::Keyboard::Key::Escape) {
          if (selected_experiment_name_.empty()) {
            running_ = false; // No experiment selected yet, quit
          } else {
            experiment_select_mode_ = false; // Return to simulation
          }
        }
      }

      // Scroll in selector
      if (const auto *scroll = event->getIf<sf::Event::MouseWheelScrolled>()) {
        int max_offset =
            std::max(0, static_cast<int>(experiment_names_.size()) - 15);
        if (scroll->delta > 0) {
          experiment_scroll_offset_ =
              std::max(0, experiment_scroll_offset_ - 3);
        } else {
          experiment_scroll_offset_ =
              std::min(max_offset, experiment_scroll_offset_ + 3);
        }
      }

      // Mouse move for hover
      if (const auto *moved = event->getIf<sf::Event::MouseMoved>()) {
        sf::Vector2f view_size = window_->getDefaultView().getSize();
        float panel_w = 400.0f;
        float panel_h = std::min(view_size.y - 80.0f, 500.0f);
        float panel_x = (view_size.x - panel_w) / 2.0f;
        float panel_y = (view_size.y - panel_h) / 2.0f;
        float list_y = panel_y + 56.0f;

        float mx = static_cast<float>(moved->position.x);
        float my = static_cast<float>(moved->position.y);

        if (mx >= panel_x + 8.0f && mx <= panel_x + panel_w - 8.0f &&
            my >= list_y && my <= panel_y + panel_h) {
          float item_h = 28.0f;
          int idx = static_cast<int>((my - list_y) / item_h) +
                    experiment_scroll_offset_;
          if (idx >= 0 && idx < static_cast<int>(experiment_names_.size())) {
            experiment_hover_index_ = idx;
          } else {
            experiment_hover_index_ = -1;
          }
        } else {
          experiment_hover_index_ = -1;
        }
      }

      // Click to select
      if (const auto *btn = event->getIf<sf::Event::MouseButtonReleased>()) {
        if (btn->button == sf::Mouse::Button::Left &&
            experiment_hover_index_ >= 0) {
          selected_experiment_name_ =
              experiment_names_[experiment_hover_index_];
          experiment_select_mode_ = false;
          experiment_selected_ = true;
          ui_state_.paused = false;
          spdlog::info("Selected experiment: {}", selected_experiment_name_);
        }
      }

      continue; // Skip normal event handling while in selector
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
          ui_state_.speed_multiplier =
              std::min(ui_state_.speed_multiplier * 2, 64);
          break;

        case sf::Keyboard::Key::Hyphen: // - key
        case sf::Keyboard::Key::Down:
        case sf::Keyboard::Key::Subtract:
          ui_state_.speed_multiplier =
              std::max(ui_state_.speed_multiplier / 2, 1);
          break;

        case sf::Keyboard::Key::R:
          ui_state_.reset_requested = true;
          break;

        case sf::Keyboard::Key::E:
          if (!experiment_names_.empty()) {
            enter_experiment_select_mode();
          }
          break;

        case sf::Keyboard::Key::S:
          if (window_) {
            sf::Texture texture(window_->getSize());
            texture.update(*window_);
            sf::Image img = texture.copyToImage();
            std::string fname = "screenshot_step" +
                                std::to_string(frame_.overlay_stats.step) +
                                ".png";
            (void)img.saveToFile(fname);
            spdlog::info("Screenshot saved: {}", fname);
          }
          break;

        case sf::Keyboard::Key::Home:
          // Reset camera to default
          zoom_level_ = 1.0f;
          camera_view_.setCenter(
              sf::Vector2f(static_cast<float>(config_.grid_size) / 2.0f,
                           static_cast<float>(config_.grid_size) / 2.0f));
          update_camera();
          break;

        default:
          break;
      }
    }

    // Mouse wheel for zoom
    if (const auto *scroll = event->getIf<sf::Event::MouseWheelScrolled>()) {
      float delta = scroll->delta;
      float factor = (delta > 0) ? 0.9f : 1.1f;
      zoom_level_ *= factor;
      zoom_level_ = std::clamp(zoom_level_, 0.1f, 10.0f);
      update_camera();
    }

    // Mouse button pressed (drag start / click select)
    if (const auto *btn = event->getIf<sf::Event::MouseButtonPressed>()) {
      if (btn->button == sf::Mouse::Button::Middle ||
          btn->button == sf::Mouse::Button::Right) {
        dragging_ = true;
        drag_start_ = sf::Vector2f(static_cast<float>(btn->position.x),
                                   static_cast<float>(btn->position.y));
        view_start_ = camera_view_.getCenter();
      }
    }

    // Mouse button released
    if (const auto *btn = event->getIf<sf::Event::MouseButtonReleased>()) {
      if (btn->button == sf::Mouse::Button::Middle ||
          btn->button == sf::Mouse::Button::Right) {
        dragging_ = false;
      }
      // Left click = select agent; store coords, apply in render() with sim
      // access
      if (btn->button == sf::Mouse::Button::Left && window_) {
        // Ignore clicks in the left column UI area
        if (btn->position.x >= static_cast<int>(ui_side_margin() + 10.0f)) {
          auto world_pos =
              window_->mapPixelToCoords(btn->position, camera_view_);
          pending_click_ = true;
          pending_click_x_ = world_pos.x;
          pending_click_y_ = world_pos.y;
        }
      }
    }

    // Mouse moved (for dragging)
    if (const auto *moved = event->getIf<sf::Event::MouseMoved>()) {
      if (dragging_ && window_) {
        sf::Vector2f current(static_cast<float>(moved->position.x),
                             static_cast<float>(moved->position.y));
        sf::Vector2f delta = drag_start_ - current;
        // Scale by zoom level
        delta.x *= zoom_level_;
        delta.y *= zoom_level_;
        camera_view_.setCenter(view_start_ + delta);
      }
    }

    // Window resized
    if (const auto *resized = event->getIf<sf::Event::Resized>()) {
      window_width_ = resized->size.x;
      window_height_ = resized->size.y;
      update_camera();
    }
  }
}

void VisualizationManager::handle_mouse_click(float world_x, float world_y) {
  // Find closest entity to click position
  float best_dist = 20.0f * zoom_level_; // click threshold in world units
  uint32_t best_agent_id = 0;

  for (const auto &agent : frame_.agents) {
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
  camera_view_.setSize(
      sf::Vector2f(view_w + 2.0f * margin, view_h + 2.0f * margin));

  // Camera stays centered - left and right UI panels are equal width (300px
  // each)
}

void VisualizationManager::update_fps(float dt) {
  if (dt > 0.0f) {
    float instant_fps = 1.0f / dt;
    current_fps_ =
        current_fps_ * (1.0f - fps_alpha_) + instant_fps * fps_alpha_;
  }
}

} // namespace moonai
