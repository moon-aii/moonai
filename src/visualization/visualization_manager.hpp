#pragma once

#include "core/config.hpp"
#include "visualization/renderer.hpp"
#include "visualization/ui_overlay.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/View.hpp>
#include <SFML/System/Clock.hpp>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace moonai {

class SimulationManager;
class EvolutionManager;

class VisualizationManager {
public:
    explicit VisualizationManager(const SimulationConfig& config);
    ~VisualizationManager();

    bool initialize();
    void render(const SimulationManager& sim, const EvolutionManager& evolution);
    bool should_close() const;
    void handle_events();

    // Update overlay stats from external sources
    void set_generation(int gen) { overlay_stats_.generation = gen; }
    void set_fitness(float best, float avg) {
        overlay_stats_.best_fitness = best;
        overlay_stats_.avg_fitness = avg;
    }
    void set_species_count(int n) { overlay_stats_.num_species = n; }
    void push_fitness(float best, float avg) { overlay_.push_fitness(best, avg); }

    // Simulation control state
    bool is_paused() const { return paused_; }
    int speed_multiplier() const { return speed_multiplier_; }
    bool should_reset() const { return reset_requested_; }
    void clear_reset() { reset_requested_ = false; }
    bool should_step() const { return step_requested_; }
    void clear_step() { step_requested_ = false; }
    bool is_fast_forward() const { return fast_forward_; }
    void clear_fast_forward() { fast_forward_ = false; }
    int selected_agent() const { return selected_agent_; }

    // Provide activation values for the selected agent's NN panel
    void set_selected_activations(
        const std::vector<float>& vals,
        const std::unordered_map<std::uint32_t, int>& idx_map);

    // Experiment selector
    void set_experiments(const std::vector<std::string>& names);
    bool in_experiment_select_mode() const { return experiment_select_mode_; }
    const std::string& selected_experiment() const { return selected_experiment_name_; }
    bool experiment_was_selected() const { return experiment_selected_; }
    void clear_experiment_selected() { experiment_selected_ = false; }
    void enter_experiment_select_mode();

private:
    void handle_mouse_click(float world_x, float world_y, const SimulationManager& sim);
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
    bool fast_forward_ = false;
    int speed_multiplier_ = 1;
    int selected_agent_ = -1;

    // Activation values for selected agent's NN visualization
    std::unordered_map<std::uint32_t, float> selected_node_activations_;

    // Camera state
    bool dragging_ = false;
    sf::Vector2f drag_start_;
    sf::Vector2f view_start_;
    float zoom_level_ = 1.0f;

    // Pending click: stored in handle_events(), applied in render() which has sim access
    bool pending_click_ = false;
    float pending_click_x_ = 0.0f;
    float pending_click_y_ = 0.0f;

    // Window dimensions
    unsigned int window_width_ = 1280;
    unsigned int window_height_ = 720;

    // Experiment selector state
    bool experiment_select_mode_ = false;
    bool experiment_selected_ = false;
    std::vector<std::string> experiment_names_;
    std::string selected_experiment_name_;
    int experiment_hover_index_ = -1;
    int experiment_scroll_offset_ = 0;
};

} // namespace moonai
