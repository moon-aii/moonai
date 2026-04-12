#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "visualization/frame_snapshot.hpp"
#include "visualization/overlay.hpp"
#include "visualization/renderer.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/View.hpp>
#include <SFML/System/Clock.hpp>

#include <cstdint>
#include <memory>

namespace moonai {

class VisualizationManager {
public:
  VisualizationManager(const SimulationConfig &config, UiState &ui_state);
  ~VisualizationManager();

  bool initialize();
  void render(FrameSnapshot frame);
  bool should_close() const;
  void handle_events();

  static constexpr float ui_side_margin() {
    return 300.0f;
  }
  static constexpr float simulation_margin() {
    return 25.0f;
  }

private:
  static constexpr unsigned int kGuiMaxFps = 120;

  void handle_mouse_click(float world_x, float world_y);
  void update_camera();

  SimulationConfig config_;
  UiState &ui_state_;
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

  bool dragging_ = false;
  sf::Vector2f drag_start_;
  sf::Vector2f view_start_;
  float zoom_level_ = 1.0f;

  bool pending_click_ = false;
  float pending_click_x_ = 0.0f;
  float pending_click_y_ = 0.0f;

  unsigned int window_width_ = 1600;
  unsigned int window_height_ = 900;
};

} // namespace moonai
