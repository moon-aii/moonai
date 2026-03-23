#pragma once

#include "evolution/genome.hpp"

#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <cstdint>

namespace moonai {

class SimulationManager;

struct OverlayStats {
    int generation = 0;
    int tick = 0;
    int alive_predators = 0;
    int alive_prey = 0;
    float best_fitness = 0.0f;
    float avg_fitness = 0.0f;
    int num_species = 0;
    float fps = 0.0f;
    int speed_multiplier = 1;
    bool paused = false;
    bool fast_forward = false;
    std::string experiment_name;

    // Selected agent info (negative = no selection)
    int selected_agent = -1;
    float selected_energy = 0.0f;
    int selected_age = 0;
    int selected_kills = 0;
    int selected_food_eaten = 0;
    float selected_fitness = 0.0f;
    int selected_genome_complexity = 0;
};

class UIOverlay {
public:
    bool initialize(const std::string& font_path = "");

    void draw(sf::RenderTarget& target, const OverlayStats& stats,
              const Genome* selected_genome = nullptr);

    bool has_font() const { return font_loaded_; }

    void push_fitness(float best, float avg);

    // Set node activation values for the selected agent's NN panel
    void set_activations(const std::unordered_map<std::uint32_t, float>& vals);

    // Experiment selector overlay
    // Returns index of clicked experiment, or -1 if none clicked
    int draw_experiment_selector(sf::RenderTarget& target,
                                  const std::vector<std::string>& names,
                                  int hover_index, int scroll_offset);

private:
    void draw_panel(sf::RenderTarget& target, float x, float y, float w, float h);
    void draw_text(sf::RenderTarget& target, const std::string& str,
                   float x, float y, unsigned int size = 14,
                   sf::Color color = sf::Color::White);
    void draw_fitness_chart(sf::RenderTarget& target);
    void draw_nn_panel(sf::RenderTarget& target, const Genome& genome);

    static constexpr int CHART_MAX_POINTS = 150;
    std::deque<float> best_history_;
    std::deque<float> avg_history_;

    std::unordered_map<std::uint32_t, float> node_activations_;

    sf::Font font_;
    bool font_loaded_ = false;
    sf::RectangleShape panel_bg_;
};

} // namespace moonai
