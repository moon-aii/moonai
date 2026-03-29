#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace moonai {

struct SimulationConfig {
  int grid_size = 6000;

  int predator_count = 480;
  int prey_count = 1920;
  int food_count = 2400;

  float predator_speed = 0.6f;
  float prey_speed = 0.66f;
  float vision_range = 360.0f;
  float interaction_range = 1.2f;
  float mate_range = 300.0f;

  float food_respawn_rate = 0.0006f;
  float energy_drain_per_step = 0.0072f;
  float energy_gain_from_kill = 96.0f;
  float energy_gain_from_food = 120.0f;
  float initial_energy = 120.0f;
  float reproduction_energy_threshold = 240.0f;
  float reproduction_energy_cost = 60.0f;
  float offspring_initial_energy = 100.0f;
  float birth_spawn_radius = 30.0f;

  float mutation_rate = 0.3f;
  float crossover_rate = 0.75f;
  float weight_mutation_power = 0.5f;
  float add_node_rate = 0.09f;
  float add_connection_rate = 0.12f;
  float delete_connection_rate = 0.02f;
  int max_hidden_nodes = 100;
  int max_steps = 0;

  float compatibility_threshold = 3.0f;
  float c1_excess = 1.0f;
  float c2_disjoint = 1.0f;
  float c3_weight = 0.4f;
  int species_update_interval_steps = 60;

  std::uint64_t seed = 0;

  std::string output_dir = "output";
  int report_interval_steps = 1000000;
};

struct ConfigError {
  std::string field;
  std::string message;
};

struct AppConfig {
  SimulationConfig sim_config;
  std::string experiment_name;
  bool headless = false;
  bool interactive = true;
  bool enable_gpu = true;
  int speed_multiplier = 1;
  std::optional<std::string> run_name_override;

  static constexpr bool cuda_compiled =
#ifdef MOONAI_ENABLE_CUDA
      true;
#else
      false;
#endif

  static constexpr bool openmp_compiled =
#ifdef MOONAI_OPENMP_ENABLED
      true;
#else
      false;
#endif

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

std::vector<ConfigError> apply_overrides(
    SimulationConfig &config,
    const std::vector<std::pair<std::string, std::string>> &overrides);

struct CLIArgs {
  std::string config_path = "config.lua";
  std::uint64_t seed_override = 0; // 0 = use config seed
  bool headless = false;
  bool verbose = false;
  bool help = false;
  bool no_gpu = false;        // --no-gpu: skip CUDA even if available
  int max_steps_override = 0; // 0 = use config value

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
