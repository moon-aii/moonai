#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace moonai {

struct SimulationConfig {
  int grid_size = 3600;

  int predator_count = 24000;
  int prey_count = 96000;
  int food_count = 240000;

  float predator_speed = 1.0f;
  float prey_speed = 1.006f;
  float vision_range = 12.0f;
  float interaction_range = 1.0f;
  float mate_range = 6.0f;

  float food_respawn_rate = 0.006f;
  float energy_drain_per_step = 0.001f;
  float energy_gain_from_kill = 0.24;
  float energy_gain_from_food = 0.24;
  float initial_energy = 0.36f;
  float max_energy = 2.0f;
  float reproduction_energy_threshold = 1.0f;
  float reproduction_energy_cost = 0.18f;
  float offspring_initial_energy = 0.36f;
  int max_age = 10000;

  float mutation_rate = 0.30f;
  float weight_mutation_power = 0.30f;
  float add_node_rate = 0.12f;
  float add_connection_rate = 0.60f;
  float delete_connection_rate = 0.00000006f;
  int max_hidden_nodes = 1200;
  int max_steps = 0;

  float compatibility_threshold = 60.0f;
  float compatibility_min_normalization = 240.0f;
  float c1_excess = 1.0f;
  float c2_disjoint = 1.0f;
  float c3_weight = 0.4f;

  int seed = 67;

  std::string output_dir = "output/experiments";
  int report_interval_steps = 1000;
};

struct ConfigError {
  std::string field;
  std::string message;
};

struct AppConfig {
  SimulationConfig sim_config;
  std::string experiment_name;
  bool headless = false;
  int speed_multiplier = 1;
  std::optional<std::string> run_name_override;

  static constexpr const char *platform =
#ifdef _WIN32
      "windows";
#else
      "linux";
#endif
};

nlohmann::json config_to_json(const SimulationConfig &config);
void save_config(const SimulationConfig &config, const std::string &filepath);

std::vector<ConfigError> validate_config(const SimulationConfig &config);

} // namespace moonai
