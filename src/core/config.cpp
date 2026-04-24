#include "core/config.hpp"

#include <fstream>
#include <spdlog/spdlog.h>

namespace moonai {

// ── JSON output (for config snapshots) ──────────────────────────────────

nlohmann::json config_to_json(const SimulationConfig &config) {
  nlohmann::json j;
  j["grid_size"] = config.grid_size;
  j["predator_count"] = config.predator_count;
  j["prey_count"] = config.prey_count;
  j["predator_speed"] = config.predator_speed;
  j["prey_speed"] = config.prey_speed;
  j["vision_range"] = config.vision_range;
  j["interaction_range"] = config.interaction_range;
  j["initial_energy"] = config.initial_energy;
  j["max_energy"] = config.max_energy;
  j["energy_drain_per_step"] = config.energy_drain_per_step;
  j["energy_gain_from_kill"] = config.energy_gain_from_kill;
  j["energy_gain_from_food"] = config.energy_gain_from_food;
  j["food_count"] = config.food_count;
  j["food_respawn_rate"] = config.food_respawn_rate;
  j["mutation_rate"] = config.mutation_rate;
  j["weight_mutation_power"] = config.weight_mutation_power;
  j["add_node_rate"] = config.add_node_rate;
  j["add_connection_rate"] = config.add_connection_rate;
  j["delete_connection_rate"] = config.delete_connection_rate;
  j["max_hidden_nodes"] = config.max_hidden_nodes;
  j["max_age"] = config.max_age;
  j["max_steps"] = config.max_steps;
  j["compatibility_threshold"] = config.compatibility_threshold;
  j["compatibility_min_normalization"] = config.compatibility_min_normalization;
  j["c1_excess"] = config.c1_excess;
  j["c2_disjoint"] = config.c2_disjoint;
  j["c3_weight"] = config.c3_weight;
  j["seed"] = config.seed;
  j["output_dir"] = config.output_dir;
  j["report_interval_steps"] = config.report_interval_steps;
  j["mate_range"] = config.mate_range;
  j["reproduction_energy_threshold"] = config.reproduction_energy_threshold;
  j["reproduction_energy_cost"] = config.reproduction_energy_cost;
  j["offspring_initial_energy"] = config.offspring_initial_energy;
  return j;
}

void save_config(const SimulationConfig &config, const std::string &filepath) {
  const nlohmann::json j = config_to_json(config);

  std::ofstream file(filepath);
  file << j.dump(4);
  spdlog::info("Config saved to '{}'.", filepath);
}

// ── Validation ──────────────────────────────────────────────────────────

std::vector<ConfigError> validate_config(const SimulationConfig &config) {
  std::vector<ConfigError> errors;

  auto check = [&](bool condition, const char *field, const char *msg) {
    if (!condition) {
      errors.push_back({field, msg});
    }
  };

  // Environment
  check(config.grid_size >= 100, "grid_size", "must be >= 100");
  check(config.grid_size <= 20000, "grid_size", "must be <= 20000");

  // Population
  check(config.predator_count >= 1, "predator_count", "must be >= 1");
  check(config.prey_count >= 1, "prey_count", "must be >= 1");
  check(config.predator_count + config.prey_count <= 1000000, "population", "total population must be <= 1000000");

  // Agent
  check(config.predator_speed > 0.0f, "predator_speed", "must be > 0");
  check(config.prey_speed > 0.0f, "prey_speed", "must be > 0");
  check(config.vision_range > 0.0f, "vision_range", "must be > 0");
  check(config.interaction_range > 0.0f, "interaction_range", "must be > 0");
  check(config.interaction_range < config.vision_range, "interaction_range", "must be less than vision_range");
  check(config.initial_energy > 0.0f, "initial_energy", "must be > 0");
  check(config.max_energy > 0.0f, "max_energy", "must be > 0");
  check(config.initial_energy <= config.max_energy, "initial_energy", "must be <= max_energy");
  check(config.energy_drain_per_step >= 0.0f, "energy_drain_per_step", "must be >= 0");

  // Food
  check(config.food_count >= 0, "food_count", "must be >= 0");
  check(config.food_respawn_rate >= 0.0f && config.food_respawn_rate <= 1.0f, "food_respawn_rate", "must be in [0, 1]");

  // Evolution rates
  check(config.mutation_rate >= 0.0f && config.mutation_rate <= 1.0f, "mutation_rate", "must be in [0, 1]");
  check(config.add_node_rate >= 0.0f && config.add_node_rate <= 1.0f, "add_node_rate", "must be in [0, 1]");
  check(config.add_connection_rate >= 0.0f && config.add_connection_rate <= 1.0f, "add_connection_rate",
        "must be in [0, 1]");
  check(config.delete_connection_rate >= 0.0f && config.delete_connection_rate <= 1.0f, "delete_connection_rate",
        "must be in [0, 1]");
  check(config.weight_mutation_power > 0.0f, "weight_mutation_power", "must be > 0");
  check(config.max_age >= 0, "max_age", "must be >= 0 (0 = infinite)");
  check(config.max_steps >= 0, "max_steps", "must be >= 0 (0 = infinite)");

  // Speciation
  check(config.compatibility_threshold > 0.0f, "compatibility_threshold", "must be > 0");
  check(config.compatibility_min_normalization >= 1.0f, "compatibility_min_normalization", "must be >= 1");

  // Simulation
  check(config.report_interval_steps >= 1, "report_interval_steps", "must be >= 1");
  check(config.mate_range > 0.0f, "mate_range", "must be > 0");
  check(config.reproduction_energy_threshold > 0.0f, "reproduction_energy_threshold", "must be > 0");
  check(config.reproduction_energy_threshold <= config.max_energy, "reproduction_energy_threshold",
        "must be <= max_energy");
  check(config.reproduction_energy_cost > 0.0f, "reproduction_energy_cost", "must be > 0");
  check(config.offspring_initial_energy > 0.0f, "offspring_initial_energy", "must be > 0");
  check(config.offspring_initial_energy <= config.max_energy, "offspring_initial_energy", "must be <= max_energy");

  return errors;
}

} // namespace moonai
