#include "visualization/visualization_manager.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/registry.hpp"
#include "simulation/simulation_manager.hpp"

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

VisualizationManager::VisualizationManager(const SimulationConfig &config)
    : config_(config) {}

VisualizationManager::~VisualizationManager() = default;

bool VisualizationManager::initialize() {
  window_ = std::make_unique<sf::RenderWindow>(
      sf::VideoMode({window_width_, window_height_}),
      "MoonAI - Predator-Prey Evolution");
  window_->setFramerateLimit(static_cast<unsigned int>(config_.target_fps));

  // Set up camera to show the simulation world
  camera_view_ =
      sf::View(sf::Vector2f(static_cast<float>(config_.grid_size) / 2.0f,
                            static_cast<float>(config_.grid_size) / 2.0f),
               sf::Vector2f(static_cast<float>(config_.grid_size),
                            static_cast<float>(config_.grid_size)));

  // Adjust view to maintain aspect ratio
  float world_aspect = static_cast<float>(config_.grid_size) /
                       static_cast<float>(config_.grid_size);
  float window_aspect = static_cast<float>(window_width_) / window_height_;

  if (window_aspect > world_aspect) {
    camera_view_.setSize(
        sf::Vector2f(static_cast<float>(config_.grid_size) * window_aspect,
                     static_cast<float>(config_.grid_size)));
  } else {
    camera_view_.setSize(
        sf::Vector2f(static_cast<float>(config_.grid_size),
                     static_cast<float>(config_.grid_size) / window_aspect));
  }

  window_->setView(camera_view_);

  overlay_.initialize();
  overlay_stats_.max_steps = config_.max_steps;

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
    paused_ = true;
  }
}

void VisualizationManager::render_ecs(const Registry &registry,
                                      const EvolutionManager &evolution,
                                      const SimulationManager &simulation,
                                      int current_step) {
  if (!window_ || !running_)
    return;

  // Update step counter
  overlay_stats_.step = current_step;

  // Experiment selector mode: render selector overlay only
  if (experiment_select_mode_) {
    window_->clear(sf::Color(20, 20, 30));
    overlay_.draw_experiment_selector(*window_, experiment_names_,
                                      experiment_hover_index_,
                                      experiment_scroll_offset_);
    window_->display();
    return;
  }

  // Apply any pending click now that we have registry access
  if (pending_click_) {
    handle_mouse_click_ecs(pending_click_x_, pending_click_y_, registry);
    pending_click_ = false;
  }

  window_->clear(sf::Color::Black);
  window_->setView(camera_view_);

  // Draw world
  renderer_.draw_background(*window_, config_.grid_size, config_.grid_size);
  renderer_.draw_grid(*window_, config_.grid_size, config_.grid_size, 500.0f);
  renderer_.draw_boundaries(*window_, config_.grid_size, config_.grid_size);

  // Draw food (ECS-based)
  renderer_.draw_food_ecs(*window_, registry);

  // Count entities and calculate stats
  int alive_predators = 0;
  int alive_prey = 0;
  int dead_count = 0;
  float pred_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float prey_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  const auto &living = registry.living_entities();
  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    const auto &vitals = registry.vitals();
    const auto &identity = registry.identity();

    if (vitals.alive[idx]) {
      if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
        alive_predators++;
      } else {
        alive_prey++;
      }

      // Calculate energy distribution
      float energy_ratio = vitals.energy[idx] / config_.initial_energy;
      energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
      int bucket = static_cast<int>(energy_ratio * 5.0f);
      bucket = std::min(bucket, 4);

      if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
        pred_dist[bucket]++;
      } else {
        prey_dist[bucket]++;
      }
    } else {
      dead_count++;
    }
  }

  // Convert to percentages
  if (alive_predators > 0) {
    for (int i = 0; i < 5; ++i) {
      pred_dist[i] /= alive_predators;
    }
  }
  if (alive_prey > 0) {
    for (int i = 0; i < 5; ++i) {
      prey_dist[i] /= alive_prey;
    }
  }
  set_energy_distribution(pred_dist, prey_dist);

  // Draw all agents using ECS
  renderer_.draw_all_agents_ecs(*window_, registry, selected_entity_);

  // Draw vision/sensor lines for selected entity (automatically shown when
  // agent is clicked)
  if (selected_entity_ != INVALID_ENTITY && registry.valid(selected_entity_)) {
    Renderer::draw_vision_range_ecs(*window_, registry, selected_entity_);
    Renderer::draw_sensor_lines_ecs(*window_, registry, selected_entity_);
  }

  // Update FPS counter
  ++frame_count_;
  float fps_elapsed = fps_clock_.getElapsedTime().asSeconds();
  if (fps_elapsed >= 0.5f) {
    overlay_stats_.fps = static_cast<float>(frame_count_) / fps_elapsed;
    frame_count_ = 0;
    fps_clock_.restart();
  }

  // Update overlay stats
  overlay_stats_.alive_predators = alive_predators;
  overlay_stats_.alive_prey = alive_prey;
  overlay_stats_.speed_multiplier = speed_multiplier_;
  overlay_stats_.paused = paused_;
  overlay_stats_.selected_agent = (selected_entity_ != INVALID_ENTITY)
                                      ? static_cast<int>(selected_entity_.index)
                                      : -1;
  overlay_stats_.experiment_name = selected_experiment_name_;

  // Update selected agent info
  if (selected_entity_ != INVALID_ENTITY && registry.valid(selected_entity_)) {
    size_t idx = registry.index_of(selected_entity_);
    const auto &vitals = registry.vitals();
    const auto &stats = registry.stats();

    overlay_stats_.selected_energy = vitals.energy[idx];
    overlay_stats_.selected_age = vitals.age[idx];
    overlay_stats_.selected_kills = stats.kills[idx];
    overlay_stats_.selected_food_eaten = stats.food_eaten[idx];

    // Get fitness and complexity from evolution manager
    auto genome = evolution.genome_for(selected_entity_);
    if (genome) {
      overlay_stats_.selected_fitness = genome->fitness();
      overlay_stats_.selected_genome_complexity = genome->complexity();
    }
  }

  // Update population chart (per step)
  overlay_.push_population(alive_predators, alive_prey);

  // Get fitness by type from evolution manager
  float best_pred = 0.0f, avg_pred = 0.0f, best_prey_f = 0.0f,
        avg_prey_f = 0.0f;
  evolution.get_fitness_by_type_ecs(registry, best_pred, avg_pred, best_prey_f,
                                    avg_prey_f);
  set_fitness_by_type(best_pred, avg_pred, best_prey_f, avg_prey_f);

  // Draw UI overlay (with selected genome for NN topology panel)
  const Genome *sel_genome = nullptr;
  if (selected_entity_ != INVALID_ENTITY) {
    sel_genome = evolution.genome_for(selected_entity_);
  }
  overlay_.set_activations(selected_node_activations_);
  overlay_.draw(*window_, overlay_stats_, sel_genome);

  window_->display();
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
          paused_ = false;
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
          paused_ = !paused_;
          break;

        case sf::Keyboard::Key::Period: // > key (step forward)
          if (paused_)
            step_requested_ = true;
          break;

        case sf::Keyboard::Key::Equal: // + key
        case sf::Keyboard::Key::Up:
        case sf::Keyboard::Key::Add:
          speed_multiplier_ = std::min(speed_multiplier_ * 2, 64);
          break;

        case sf::Keyboard::Key::Hyphen: // - key
        case sf::Keyboard::Key::Down:
        case sf::Keyboard::Key::Subtract:
          speed_multiplier_ = std::max(speed_multiplier_ / 2, 1);
          break;

        case sf::Keyboard::Key::R:
          reset_requested_ = true;
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
                                std::to_string(overlay_stats_.step) + ".png";
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
        if (btn->position.x >= static_cast<int>(left_column_width() + 10.0f)) {
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

void VisualizationManager::handle_mouse_click_ecs(float world_x, float world_y,
                                                  const Registry &registry) {
  // Find closest entity to click position
  float best_dist = 20.0f * zoom_level_; // click threshold in world units
  Entity best_entity = INVALID_ENTITY;

  const auto &living = registry.living_entities();
  for (Entity entity : living) {
    size_t idx = registry.index_of(entity);
    const auto &vitals = registry.vitals();

    if (!vitals.alive[idx]) {
      continue;
    }

    const auto &positions = registry.positions();
    float dx = positions.x[idx] - world_x;
    float dy = positions.y[idx] - world_y;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist < best_dist) {
      best_dist = dist;
      best_entity = entity;
    }
  }

  selected_entity_ = best_entity;
}

void VisualizationManager::set_selected_activations(
    const std::vector<float> &vals,
    const std::unordered_map<std::uint32_t, int> &idx_map) {
  selected_node_activations_.clear();
  // idx_map: node_id -> array index; invert to fill {node_id ->
  // activation_value}
  for (const auto &[node_id, idx] : idx_map) {
    if (idx >= 0 && idx < static_cast<int>(vals.size())) {
      selected_node_activations_[node_id] = vals[idx];
    }
  }
}

void VisualizationManager::update_camera() {
  if (!window_)
    return;

  float world_w = static_cast<float>(config_.grid_size);
  float world_h = static_cast<float>(config_.grid_size);

  float window_aspect = static_cast<float>(window_width_) / window_height_;
  float world_aspect = world_w / world_h;

  float view_w, view_h;
  if (window_aspect > world_aspect) {
    view_h = world_h * zoom_level_;
    view_w = view_h * window_aspect;
  } else {
    view_w = world_w * zoom_level_;
    view_h = view_w / window_aspect;
  }

  camera_view_.setSize(sf::Vector2f(view_w, view_h));

  // Shift camera to the right to account for left column UI
  sf::Vector2f current_center = camera_view_.getCenter();
  camera_view_.setCenter(sf::Vector2f(
      current_center.x + left_column_width() / 2.0f, current_center.y));
}

} // namespace moonai
