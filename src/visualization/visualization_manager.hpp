#pragma once

#include "core/config.hpp"
#include "simulation/entity.hpp"
#include "visualization/renderer.hpp"
#include "visualization/ui_overlay.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/View.hpp>
#include <SFML/System/Clock.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonai {

class Registry;
class EvolutionManager;
class SimulationManager;

class VisualizationManager {
public:
  explicit VisualizationManager(const SimulationConfig &config);
  ~VisualizationManager();

  bool initialize();
  void render_ecs(const Registry &registry, const EvolutionManager &evolution,
                  const SimulationManager &simulation, int current_step);
  bool should_close() const;
  void handle_events();

  void set_fitness(float best, float avg) {
    overlay_stats_.best_fitness = best;
    overlay_stats_.avg_fitness = avg;
  }
  void set_species_count(int n) {
    overlay_stats_.num_species = n;
  }
  void push_fitness(float best, float avg) {
    overlay_.push_fitness(best, avg);
  }

  bool is_paused() const {
    return paused_;
  }
  int speed_multiplier() const {
    return speed_multiplier_;
  }
  bool should_reset() const {
    return reset_requested_;
  }
  void clear_reset() {
    reset_requested_ = false;
  }
  bool should_step() const {
    return step_requested_;
  }
  void clear_step() {
    step_requested_ = false;
  }
  Entity selected_entity() const {
    return selected_entity_;
  }

  void set_selected_activations(
      const std::vector<float> &vals,
      const std::unordered_map<std::uint32_t, int> &idx_map);

  void update_population_chart(int predators, int prey) {
    overlay_.push_population(predators, prey);
  }

  void set_fitness_by_type(float best_pred, float avg_pred, float best_prey,
                           float avg_prey) {
    overlay_stats_.best_predator_fitness = best_pred;
    overlay_stats_.avg_predator_fitness = avg_pred;
    overlay_stats_.best_prey_fitness = best_prey;
    overlay_stats_.avg_prey_fitness = avg_prey;
  }

  void set_energy_distribution(const float pred_dist[5],
                               const float prey_dist[5]) {
    for (int i = 0; i < 5; ++i) {
      overlay_stats_.predator_energy_dist[i] = pred_dist[i];
      overlay_stats_.prey_energy_dist[i] = prey_dist[i];
    }
  }

  void set_event_counts(int kills, int food, int births, int deaths) {
    overlay_stats_.kills_this_step = kills;
    overlay_stats_.food_eaten_this_step = food;
    overlay_stats_.births_this_step = births;
    overlay_stats_.deaths_this_step = deaths;
  }

  static constexpr float left_column_width() {
    return 260.0f;
  }

  void set_experiments(const std::vector<std::string> &names);
  bool in_experiment_select_mode() const {
    return experiment_select_mode_;
  }
  const std::string &selected_experiment() const {
    return selected_experiment_name_;
  }
  bool experiment_was_selected() const {
    return experiment_selected_;
  }
  void clear_experiment_selected() {
    experiment_selected_ = false;
  }
  void enter_experiment_select_mode();

private:
  void handle_mouse_click_ecs(float world_x, float world_y,
                              const Registry &registry);
  void update_camera();

  SimulationConfig config_;
  std::unique_ptr<sf::RenderWindow> window_;
  sf::View camera_view_;
  Renderer renderer_;
  UIOverlay overlay_;
  OverlayStats overlay_stats_;
  sf::Clock frame_clock_;
  sf::Clock fps_clock_;
  int frame_count_ = 0;

  bool running_ = false;
  bool paused_ = false;
  bool reset_requested_ = false;
  bool step_requested_ = false;
  int speed_multiplier_ = 1;
  Entity selected_entity_ = INVALID_ENTITY;

  std::unordered_map<std::uint32_t, float> selected_node_activations_;

  bool dragging_ = false;
  sf::Vector2f drag_start_;
  sf::Vector2f view_start_;
  float zoom_level_ = 1.0f;

  bool pending_click_ = false;
  float pending_click_x_ = 0.0f;
  float pending_click_y_ = 0.0f;

  unsigned int window_width_ = 1600;
  unsigned int window_height_ = 900;

  bool experiment_select_mode_ = false;
  bool experiment_selected_ = false;
  std::vector<std::string> experiment_names_;
  std::string selected_experiment_name_;
  int experiment_hover_index_ = -1;
  int experiment_scroll_offset_ = 0;
};

} // namespace moonai
