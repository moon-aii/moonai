#pragma once

#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace moonai {

struct SimulationConfig {
  int grid_size = 3000;

  int predator_count = 600;
  int prey_count = 2400;

  float predator_speed = 0.06f;
  float prey_speed = 0.054f;
  float vision_range = 120.0f;
  float attack_range = 6.0f;
  float initial_energy = 120.0f;
  float energy_drain_per_step = 0.006f;
  float energy_gain_from_kill = 36.0f;
  float energy_gain_from_food = 48.0f;
  float food_pickup_range = 6.0f;

  int food_count = 3000;
  float food_respawn_rate = 0.006f;

  float mutation_rate = 0.3f;
  float crossover_rate = 0.75f;
  float weight_mutation_power = 0.5f;
  float add_node_rate = 0.03f;
  float add_connection_rate = 0.05f;
  float delete_connection_rate = 0.01f;
  int max_hidden_nodes = 100;
  int max_steps = 0;

  float compatibility_threshold = 3.0f;
  float c1_excess = 1.0f;
  float c2_disjoint = 1.0f;
  float c3_weight = 0.4f;
  int species_update_interval_steps = 60;

  std::uint64_t seed = 0;

  std::string output_dir = "output";
  int report_interval_steps = 1000;

  float mate_range = 60.0f;
  float reproduction_energy_threshold = 180.0f;
  float reproduction_energy_cost = 60.0f;
  float offspring_initial_energy = 100.0f;
  int min_reproductive_age_steps = 120;
  int reproduction_cooldown_steps = 300;
  float birth_spawn_radius = 8.0f;

  float fitness_survival_weight = 1.0f;
  float fitness_kill_weight = 5.0f;
  float fitness_energy_weight = 0.5f;
  float fitness_distance_weight = 0.1f;
  float complexity_penalty_weight = 0.01f;

  std::string activation_function = "sigmoid";
};

struct ConfigError {
  std::string field;
  std::string message;
};

nlohmann::json config_to_json(const SimulationConfig &config);
std::string fingerprint_config(const SimulationConfig &config);
void save_config(const SimulationConfig &config, const std::string &filepath);

std::vector<ConfigError> validate_config(const SimulationConfig &config);

std::vector<ConfigError> apply_overrides(
    SimulationConfig &config,
    const std::vector<std::pair<std::string, std::string>> &overrides);

void apply_overrides_float(SimulationConfig &config,
                           const std::map<std::string, float> &overrides);

struct CLIArgs {
  std::string config_path = "config.lua";
  std::uint64_t seed_override = 0; // 0 = use config seed
  bool headless = false;
  bool verbose = false;
  bool help = false;
  bool no_gpu = false;          // --no-gpu: skip CUDA even if available
  int max_steps_override = 0;   // 0 = use config value
  std::string resume_path = ""; // path to checkpoint JSON; "" = fresh start
  int checkpoint_interval = 0;  // 0 = disabled; N = save every N steps
  std::string compare_a = "";   // --compare: path to first genome JSON
  std::string compare_b = "";   // --compare: path to second genome JSON

  // Lua config orchestration
  std::string experiment_name =
      "";                        // --experiment: select one from multi-config
  bool run_all = false;          // --all: run all experiments sequentially
  bool list_experiments = false; // --list: list experiment names and exit
  std::string run_name = "";     // --name: override output directory name
  bool validate_only = false;    // --validate: check config and exit

  // --set key=value overrides
  std::vector<std::pair<std::string, std::string>> overrides;
};

CLIArgs parse_args(int argc, const char *argv[]);
void print_usage(const char *program_name);

} // namespace moonai
