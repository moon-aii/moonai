#pragma once

#include "evolution/genome.hpp"
#include "visualization/visual_constants.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonai {

class SimulationManager;

struct OverlayStats {
  int step = 0;
  int max_steps = 1500;
  int alive_predators = 0;
  int alive_prey = 0;
  int active_food = 0;
  float best_fitness = 0.0f;
  float avg_fitness = 0.0f;
  int num_species = 0;
  float fps = 0.0f;
  int speed_multiplier = 1;
  bool paused = false;
  std::string experiment_name;

  // Selected agent info (negative = no selection)
  int selected_agent = -1;
  float selected_energy = 0.0f;
  int selected_age = 0;
  int selected_kills = 0;
  int selected_food_eaten = 0;
  float selected_fitness = 0.0f;
  int selected_genome_complexity = 0;

  // Fitness by type
  float best_predator_fitness = 0.0f;
  float avg_predator_fitness = 0.0f;
  float best_prey_fitness = 0.0f;
  float avg_prey_fitness = 0.0f;

  // Energy distribution (5 buckets: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
  // Percentage of agents in each energy bucket (0.0 to 1.0)
  float predator_energy_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float prey_energy_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // Event counts for the whole run
  int total_kills = 0;
  int total_food_eaten = 0;
  int total_births = 0;
  int total_deaths = 0;
};

class UIOverlay {
public:
  bool initialize(const std::string &font_path = "");

  void draw(sf::RenderTarget &target, const OverlayStats &stats,
            const Genome *selected_genome = nullptr);

  bool has_font() const {
    return font_loaded_;
  }

  void push_fitness(float best, float avg);

  // Set node activation values for the selected agent's NN panel
  void set_activations(const std::unordered_map<std::uint32_t, float> &vals);

  void push_population(int predators, int prey, int food);

  void push_species(int count);

  int draw_experiment_selector(sf::RenderTarget &target,
                               const std::vector<std::string> &names,
                               int hover_index, int scroll_offset);

private:
  void draw_panel(sf::RenderTarget &target, float x, float y, float w, float h);
  void draw_text(sf::RenderTarget &target, const std::string &str, float x,
                 float y, unsigned int size = 14,
                 sf::Color color = sf::Color::White);
  void draw_fitness_chart(sf::RenderTarget &target);
  void draw_nn_panel(sf::RenderTarget &target, const Genome &genome);

  // Left column panels
  void draw_left_column(sf::RenderTarget &target, const OverlayStats &stats);
  void draw_stats_panel(sf::RenderTarget &target, const OverlayStats &stats,
                        float x, float y);
  void draw_population_chart(sf::RenderTarget &target, float x, float y,
                             float w, float h);
  void draw_fitness_by_type(sf::RenderTarget &target, const OverlayStats &stats,
                            float x, float y, float w, float h);
  void draw_energy_distribution(sf::RenderTarget &target,
                                const OverlayStats &stats, float x, float y,
                                float w, float h);
  void draw_stats_widget(sf::RenderTarget &target, const OverlayStats &stats,
                         float x, float y, float w, float h);

  std::deque<float> best_history_;
  std::deque<float> avg_history_;

  // Population history: tuple of {predators, prey, food}
  std::deque<std::tuple<int, int, int>> population_history_;
  std::deque<int> species_history_;

  std::unordered_map<std::uint32_t, float> node_activations_;

  sf::Font font_;
  bool font_loaded_ = false;
  sf::RectangleShape panel_bg_;
};

} // namespace moonai
