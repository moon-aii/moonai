#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace moonai {

enum class BoundaryMode { Clamp, Wrap };

struct SimulationConfig {
    // ── Environment ─────────────────────────────────────────────────────
    int grid_width  = 4300;          // World width in simulation units [100, 20000]
    int grid_height = 2400;          // World height in simulation units [100, 20000]
    BoundaryMode boundary_mode = BoundaryMode::Wrap;  // "wrap" or "clamp"

    // ── Population ──────────────────────────────────────────────────────
    int predator_count = 500;        // Number of predator agents [1, ...]
    int prey_count     = 1500;       // Number of prey agents [1, ...]

    // ── Agent Parameters ────────────────────────────────────────────────
    float predator_speed = 4.0f;     // Predator max speed (units/tick) [> 0]
    float prey_speed     = 4.5f;     // Prey max speed (units/tick); prey slightly faster [> 0]
    float vision_range   = 200.0f;   // Radius each agent can see (units) [> 0]
    float attack_range   = 20.0f;    // Predator kill radius (units) [> 0, < vision_range]
    float initial_energy = 150.0f;   // Starting energy; death at <= 0 [> 0]
    float energy_drain_per_tick  = 0.08f;  // Fixed energy cost per tick (living cost) [>= 0]
    float energy_gain_from_kill  = 60.0f;  // Energy predator gains per successful kill
    float energy_gain_from_food  = 40.0f;  // Energy prey gains per food pellet eaten
    float food_pickup_range      = 12.0f;  // Prey food detection radius (units) [> 0]

    // ── Food / Resources ────────────────────────────────────────────────
    int   food_count        = 2500;   // Food pellets to spawn at initialization [>= 0]
    float food_respawn_rate = 0.02f;  // P(respawn per empty slot per tick) ∈ [0, 1]

    // ── NEAT - Mutation (probabilities applied independently each generation) ──
    float mutation_rate         = 0.3f;   // P(weight mutation per genome) ∈ [0, 1]
    float crossover_rate        = 0.75f;  // P(use crossover vs clone) ∈ [0, 1]
    float weight_mutation_power = 0.5f;   // Std dev for Gaussian weight perturbation [> 0]
    float add_node_rate         = 0.03f;  // P(add node mutation per genome) ∈ [0, 1]
    float add_connection_rate   = 0.05f;  // P(add connection mutation per genome) ∈ [0, 1]
    float delete_connection_rate = 0.01f; // P(delete connection mutation per genome) ∈ [0, 1]
    int   max_hidden_nodes      = 100;    // Max hidden nodes per genome; 0 = unlimited [>= 0]
    int   generation_ticks      = 1500;   // Simulation steps per generation [>= 10]
    int   max_generations       = 0;      // 0 = run indefinitely; otherwise stop after N

    // ── NEAT - Speciation (Stanley 2002, Section 3.3) ───────────────────
    float compatibility_threshold = 3.0f;  // δ_t: genomes within this distance → same species [> 0]
    float c1_excess   = 1.0f;   // Coefficient for excess genes in δ formula
    float c2_disjoint = 1.0f;   // Coefficient for disjoint genes in δ formula
    float c3_weight   = 0.4f;   // Coefficient for avg weight diff in δ formula
    int   stagnation_limit = 20; // Generations without improvement before species is culled [>= 1]

    // ── Simulation ──────────────────────────────────────────────────────
    int           target_fps = 60;   // Render/physics framerate cap (also sets dt) [1, 1000]
    std::uint64_t seed       = 0;    // Master RNG seed; 0 = random from clock

    // ── Data Logging ────────────────────────────────────────────────────
    std::string output_dir  = "output";  // Directory for CSV/JSON run data
    int         log_interval = 1;        // Log stats every N generations [>= 1]

    // ── Fitness Weights (linear combination forming genome fitness) ──────
    float fitness_survival_weight   = 1.0f;   // Reward for surviving (0 → 1 fraction of gen)
    float fitness_kill_weight       = 5.0f;   // Reward per kill (predator) or food (prey)
    float fitness_energy_weight     = 0.5f;   // Reward for remaining energy ratio
    float fitness_distance_weight   = 0.1f;   // Reserved for future distance-traveled metric
    float complexity_penalty_weight = 0.01f;  // Fitness penalty per node+connection [>= 0]

    // ── Neural Network ───────────────────────────────────────────────────
    std::string activation_function = "sigmoid";  // Activation fn: "sigmoid", "tanh", "relu"

    // ── Per-Tick Logging (optional, high-volume) ─────────────────────────
    bool tick_log_enabled  = false;  // Enable per-tick agent state logging
    int  tick_log_interval = 10;     // Log agent states every N ticks [>= 1]
};

struct ConfigError {
    std::string field;
    std::string message;
};

// ── Lua config loading ──────────────────────────────────────────────────
// Load named configs from a Lua file.
// The file must return a table of the form: { name = { ...params... }, ... }
// A single-entry file (e.g. { default = {...} }) is auto-selected without --experiment.
std::map<std::string, SimulationConfig> load_all_configs_lua(const std::string& filepath);

// ── JSON output (for config snapshots in output dirs) ───────────────────
void save_config(const SimulationConfig& config, const std::string& filepath);

// ── Validation ──────────────────────────────────────────────────────────
std::vector<ConfigError> validate_config(const SimulationConfig& config);

// ── CLI override ────────────────────────────────────────────────────────
// Apply --set key=value overrides to a config. Returns errors for unknown keys.
std::vector<ConfigError> apply_overrides(
    SimulationConfig& config,
    const std::vector<std::pair<std::string, std::string>>& overrides);

// Apply float-valued overrides from Lua hooks (e.g. { mutation_rate = 0.5 }).
// Only applies to recognized float/int fields. Logs warnings for unknown keys.
void apply_overrides_float(SimulationConfig& config,
                           const std::map<std::string, float>& overrides);

// ── CLI ─────────────────────────────────────────────────────────────────
struct CLIArgs {
    std::string config_path = "config.lua";
    std::uint64_t seed_override = 0;  // 0 = use config seed
    bool headless = false;
    bool verbose = false;
    bool help = false;
    bool no_gpu = false;               // --no-gpu: skip CUDA even if available
    int max_generations_override = 0;  // 0 = use config value
    std::string resume_path = "";      // path to checkpoint JSON; "" = fresh start
    int checkpoint_interval = 0;       // 0 = disabled; N = save every N generations
    std::string compare_a = "";        // --compare: path to first genome JSON
    std::string compare_b = "";        // --compare: path to second genome JSON

    // Lua config orchestration
    std::string experiment_name = "";  // --experiment: select one from multi-config
    bool run_all = false;              // --all: run all experiments sequentially
    bool list_experiments = false;     // --list: list experiment names and exit
    std::string run_name = "";         // --name: override output directory name
    bool validate_only = false;        // --validate: check config and exit

    // --set key=value overrides
    std::vector<std::pair<std::string, std::string>> overrides;
};

CLIArgs parse_args(int argc, char* argv[]);
void print_usage(const char* program_name);

} // namespace moonai
