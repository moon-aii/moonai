#include "core/config.hpp"

#include <fstream>
#include <cstdio>
#include <spdlog/spdlog.h>

namespace moonai {

namespace {

template<typename T>
void json_get(const nlohmann::json& j, const char* key, T& field) {
    if (j.contains(key)) {
        field = j[key].get<T>();
    }
}

void json_get_boundary(const nlohmann::json& j, const char* key, BoundaryMode& field) {
    if (j.contains(key)) {
        std::string val = j[key].get<std::string>();
        if (val == "wrap") field = BoundaryMode::Wrap;
        else if (val == "clamp") field = BoundaryMode::Clamp;
    }
}

} // anonymous namespace

SimulationConfig load_config(const std::string& filepath) {
    SimulationConfig config;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        spdlog::warn("Config file '{}' not found, using defaults.", filepath);
        return config;
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::parse_error& e) {
        spdlog::error("Failed to parse config '{}': {}", filepath, e.what());
        return config;
    }

    json_get(j, "grid_width", config.grid_width);
    json_get(j, "grid_height", config.grid_height);
    json_get_boundary(j, "boundary_mode", config.boundary_mode);
    json_get(j, "predator_count", config.predator_count);
    json_get(j, "prey_count", config.prey_count);
    json_get(j, "predator_speed", config.predator_speed);
    json_get(j, "prey_speed", config.prey_speed);
    json_get(j, "vision_range", config.vision_range);
    json_get(j, "attack_range", config.attack_range);
    json_get(j, "initial_energy", config.initial_energy);
    json_get(j, "energy_drain_per_tick", config.energy_drain_per_tick);
    json_get(j, "energy_gain_from_kill", config.energy_gain_from_kill);
    json_get(j, "energy_gain_from_food", config.energy_gain_from_food);
    json_get(j, "food_pickup_range", config.food_pickup_range);
    json_get(j, "food_count", config.food_count);
    json_get(j, "food_respawn_rate", config.food_respawn_rate);
    json_get(j, "mutation_rate", config.mutation_rate);
    json_get(j, "crossover_rate", config.crossover_rate);
    json_get(j, "weight_mutation_power", config.weight_mutation_power);
    json_get(j, "add_node_rate", config.add_node_rate);
    json_get(j, "add_connection_rate", config.add_connection_rate);
    json_get(j, "delete_connection_rate", config.delete_connection_rate);
    json_get(j, "max_hidden_nodes", config.max_hidden_nodes);
    json_get(j, "generation_ticks", config.generation_ticks);
    json_get(j, "max_generations", config.max_generations);
    json_get(j, "compatibility_threshold", config.compatibility_threshold);
    json_get(j, "c1_excess", config.c1_excess);
    json_get(j, "c2_disjoint", config.c2_disjoint);
    json_get(j, "c3_weight", config.c3_weight);
    json_get(j, "stagnation_limit", config.stagnation_limit);
    json_get(j, "target_fps", config.target_fps);
    json_get(j, "seed", config.seed);
    json_get(j, "output_dir", config.output_dir);
    json_get(j, "log_interval", config.log_interval);
    json_get(j, "fitness_survival_weight", config.fitness_survival_weight);
    json_get(j, "fitness_kill_weight", config.fitness_kill_weight);
    json_get(j, "fitness_energy_weight", config.fitness_energy_weight);
    json_get(j, "fitness_distance_weight", config.fitness_distance_weight);
    json_get(j, "complexity_penalty_weight", config.complexity_penalty_weight);
    json_get(j, "activation_function", config.activation_function);
    json_get(j, "tick_log_enabled", config.tick_log_enabled);
    json_get(j, "tick_log_interval", config.tick_log_interval);

    spdlog::info("Config loaded from '{}'.", filepath);
    return config;
}

void save_config(const SimulationConfig& config, const std::string& filepath) {
    nlohmann::json j;

    j["grid_width"] = config.grid_width;
    j["grid_height"] = config.grid_height;
    j["boundary_mode"] = (config.boundary_mode == BoundaryMode::Wrap) ? "wrap" : "clamp";
    j["predator_count"] = config.predator_count;
    j["prey_count"] = config.prey_count;
    j["predator_speed"] = config.predator_speed;
    j["prey_speed"] = config.prey_speed;
    j["vision_range"] = config.vision_range;
    j["attack_range"] = config.attack_range;
    j["initial_energy"] = config.initial_energy;
    j["energy_drain_per_tick"] = config.energy_drain_per_tick;
    j["energy_gain_from_kill"] = config.energy_gain_from_kill;
    j["energy_gain_from_food"] = config.energy_gain_from_food;
    j["food_pickup_range"] = config.food_pickup_range;
    j["food_count"] = config.food_count;
    j["food_respawn_rate"] = config.food_respawn_rate;
    j["mutation_rate"] = config.mutation_rate;
    j["crossover_rate"] = config.crossover_rate;
    j["weight_mutation_power"] = config.weight_mutation_power;
    j["add_node_rate"] = config.add_node_rate;
    j["add_connection_rate"] = config.add_connection_rate;
    j["delete_connection_rate"] = config.delete_connection_rate;
    j["max_hidden_nodes"] = config.max_hidden_nodes;
    j["generation_ticks"] = config.generation_ticks;
    j["max_generations"] = config.max_generations;
    j["compatibility_threshold"] = config.compatibility_threshold;
    j["c1_excess"] = config.c1_excess;
    j["c2_disjoint"] = config.c2_disjoint;
    j["c3_weight"] = config.c3_weight;
    j["stagnation_limit"] = config.stagnation_limit;
    j["target_fps"] = config.target_fps;
    j["seed"] = config.seed;
    j["output_dir"] = config.output_dir;
    j["log_interval"] = config.log_interval;
    j["fitness_survival_weight"] = config.fitness_survival_weight;
    j["fitness_kill_weight"] = config.fitness_kill_weight;
    j["fitness_energy_weight"] = config.fitness_energy_weight;
    j["fitness_distance_weight"] = config.fitness_distance_weight;
    j["complexity_penalty_weight"] = config.complexity_penalty_weight;
    j["activation_function"] = config.activation_function;
    j["tick_log_enabled"] = config.tick_log_enabled;
    j["tick_log_interval"] = config.tick_log_interval;

    std::ofstream file(filepath);
    file << j.dump(4);
    spdlog::info("Config saved to '{}'.", filepath);
}

std::vector<ConfigError> validate_config(const SimulationConfig& config) {
    std::vector<ConfigError> errors;

    auto check = [&](bool condition, const char* field, const char* msg) {
        if (!condition) {
            errors.push_back({field, msg});
        }
    };

    // Environment
    check(config.grid_width >= 100, "grid_width", "must be >= 100");
    check(config.grid_width <= 10000, "grid_width", "must be <= 10000");
    check(config.grid_height >= 100, "grid_height", "must be >= 100");
    check(config.grid_height <= 10000, "grid_height", "must be <= 10000");

    // Population
    check(config.predator_count >= 1, "predator_count", "must be >= 1");
    check(config.prey_count >= 1, "prey_count", "must be >= 1");
    check(config.predator_count + config.prey_count <= 10000,
          "population", "total population must be <= 10000");

    // Agent
    check(config.predator_speed > 0.0f, "predator_speed", "must be > 0");
    check(config.prey_speed > 0.0f, "prey_speed", "must be > 0");
    check(config.vision_range > 0.0f, "vision_range", "must be > 0");
    check(config.attack_range > 0.0f, "attack_range", "must be > 0");
    check(config.attack_range < config.vision_range,
          "attack_range", "must be less than vision_range");
    check(config.initial_energy > 0.0f, "initial_energy", "must be > 0");
    check(config.energy_drain_per_tick >= 0.0f, "energy_drain_per_tick", "must be >= 0");

    // Food
    check(config.food_count >= 0, "food_count", "must be >= 0");
    check(config.food_respawn_rate >= 0.0f && config.food_respawn_rate <= 1.0f,
          "food_respawn_rate", "must be in [0, 1]");

    // Evolution rates
    check(config.mutation_rate >= 0.0f && config.mutation_rate <= 1.0f,
          "mutation_rate", "must be in [0, 1]");
    check(config.crossover_rate >= 0.0f && config.crossover_rate <= 1.0f,
          "crossover_rate", "must be in [0, 1]");
    check(config.add_node_rate >= 0.0f && config.add_node_rate <= 1.0f,
          "add_node_rate", "must be in [0, 1]");
    check(config.add_connection_rate >= 0.0f && config.add_connection_rate <= 1.0f,
          "add_connection_rate", "must be in [0, 1]");
    check(config.delete_connection_rate >= 0.0f && config.delete_connection_rate <= 1.0f,
          "delete_connection_rate", "must be in [0, 1]");
    check(config.weight_mutation_power > 0.0f, "weight_mutation_power", "must be > 0");
    check(config.generation_ticks >= 10, "generation_ticks", "must be >= 10");
    check(config.max_generations >= 0, "max_generations", "must be >= 0 (0 = infinite)");

    // Speciation
    check(config.compatibility_threshold > 0.0f,
          "compatibility_threshold", "must be > 0");
    check(config.stagnation_limit >= 1, "stagnation_limit", "must be >= 1");

    // Simulation
    check(config.target_fps >= 1 && config.target_fps <= 1000,
          "target_fps", "must be in [1, 1000]");
    check(config.log_interval >= 1, "log_interval", "must be >= 1");
    check(config.tick_log_interval >= 1, "tick_log_interval", "must be >= 1");

    return errors;
}

CLIArgs parse_args(int argc, char* argv[]) {
    CLIArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if (arg == "--headless") {
            args.headless = true;
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            try {
                args.seed_override = std::stoull(argv[++i]);
            } catch (const std::exception&) {
                std::fprintf(stderr, "Invalid seed value '%s'\n", argv[i]);
                args.help = true;
            }
        } else if ((arg == "-g" || arg == "--generations") && i + 1 < argc) {
            try {
                args.max_generations_override = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                std::fprintf(stderr, "Invalid generations value '%s'\n", argv[i]);
                args.help = true;
            }
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            args.config_path = argv[++i];
        } else if (arg == "--resume" && i + 1 < argc) {
            args.resume_path = argv[++i];
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            try {
                args.checkpoint_interval = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                std::fprintf(stderr, "Invalid checkpoint interval '%s'\n", argv[i]);
                args.help = true;
            }
        } else if (arg == "--no-gpu") {
            args.no_gpu = true;
        } else if (arg == "--compare" && i + 2 < argc) {
            args.compare_a = argv[++i];
            args.compare_b = argv[++i];
        } else if (arg[0] != '-') {
            // Positional argument: config path
            args.config_path = arg;
        } else {
            spdlog::warn("Unknown argument: {}", arg);
        }
    }

    return args;
}

void print_usage(const char* program_name) {
    fmt::print(
        "MoonAI - Predator-Prey Evolutionary Simulation\n"
        "\n"
        "Usage: {} [OPTIONS] [config_path]\n"
        "\n"
        "Options:\n"
        "  -c, --config <path>       Path to config JSON (default: config/default_config.json)\n"
        "  -s, --seed <number>       Override random seed\n"
        "  -g, --generations <n>     Override max generations (0 = infinite)\n"
        "      --headless            Run without visualization\n"
        "  -v, --verbose             Enable debug logging\n"
        "      --resume <path>       Resume from a checkpoint JSON file\n"
        "      --checkpoint <n>      Save checkpoint every N generations (0 = disabled)\n"
        "      --compare <a> <b>     Print structural diff between two genome JSON files\n"
        "      --no-gpu              Disable CUDA GPU acceleration (use CPU path)"
#ifdef MOONAI_ENABLE_CUDA
        " (cuda compiled in)\n"
#else
        " (no-cuda build: always CPU)\n"
#endif
        "  -h, --help                Show this help message\n",
        program_name
    );
}

} // namespace moonai
