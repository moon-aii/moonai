#pragma once

#include "core/config.hpp"
#include "core/types.hpp"
#include "simulation/entity.hpp"
#include "visualization/overlay.hpp"
#include "visualization/renderer.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/View.hpp>
#include <SFML/System/Clock.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonai {

struct FrameSnapshot {
  int world_width = 0;
  int world_height = 0;
  std::vector<RenderFood> foods;
  std::vector<RenderAgent> agents;
  bool has_selected_vision = false;
  uint32_t selected_agent_id = 0;
  Vec2 selected_position;
  float selected_vision_range = 0.0f;
  std::vector<RenderLine> sensor_lines;
  OverlayStats overlay_stats;
  const Genome *selected_genome = nullptr;
  std::unordered_map<std::uint32_t, float> selected_node_activations;
};

class VisualizationManager {
public:
  explicit VisualizationManager(const SimulationConfig &config);
  ~VisualizationManager();

  bool initialize();
  void render(FrameSnapshot frame);
  bool should_close() const;
  void handle_events();

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
  uint32_t selected_agent_id() const {
    return selected_agent_id_;
  }

  static constexpr float ui_side_margin() {
    return 300.0f;
  }
  static constexpr float simulation_margin() {
    return 25.0f;
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
  static constexpr unsigned int kGuiMaxFps = 360;

  void handle_mouse_click(float world_x, float world_y);
  void update_camera();

  SimulationConfig config_;
  std::unique_ptr<sf::RenderWindow> window_;
  sf::View camera_view_;
  Renderer renderer_;
  UIOverlay overlay_;
  FrameSnapshot frame_;
  sf::Clock frame_clock_;
  int last_chart_step_ = -1;

  float current_fps_ = 60.0f;
  static constexpr float fps_alpha_ = 0.1f;

  void update_fps(float dt);

  bool running_ = false;
  bool paused_ = false;
  bool reset_requested_ = false;
  bool step_requested_ = false;
  int speed_multiplier_ = 1;
  uint32_t selected_agent_id_ = 0;

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
