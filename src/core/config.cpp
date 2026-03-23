#include "core/config.hpp"

#include <fstream>
#include <cstdio>
#include <spdlog/spdlog.h>

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace moonai {

namespace {

// Read a Lua table field into a C++ variable (silently skips if absent)
template<typename T>
void lua_get(const sol::table& tbl, const char* key, T& field) {
    auto val = tbl[key];
    if (val.valid()) {
        field = val.get<T>();
    }
}

void lua_get_boundary(const sol::table& tbl, const char* key, BoundaryMode& field) {
    auto val = tbl[key];
    if (val.valid()) {
        std::string s = val.get<std::string>();
        if (s == "wrap")  field = BoundaryMode::Wrap;
        else if (s == "clamp") field = BoundaryMode::Clamp;
    }
}

void lua_get_bool(const sol::table& tbl, const char* key, bool& field) {
    auto val = tbl[key];
    if (val.valid()) {
        field = val.get<bool>();
    }
}

void lua_get_uint64(const sol::table& tbl, const char* key, std::uint64_t& field) {
    auto val = tbl[key];
    if (val.valid()) {
        field = static_cast<std::uint64_t>(val.get<double>());
    }
}

// Populate a SimulationConfig from a Lua table
SimulationConfig table_to_config(const sol::table& tbl) {
    SimulationConfig config;

    lua_get(tbl, "grid_width", config.grid_width);
    lua_get(tbl, "grid_height", config.grid_height);
    lua_get_boundary(tbl, "boundary_mode", config.boundary_mode);
    lua_get(tbl, "predator_count", config.predator_count);
    lua_get(tbl, "prey_count", config.prey_count);
    lua_get(tbl, "predator_speed", config.predator_speed);
    lua_get(tbl, "prey_speed", config.prey_speed);
    lua_get(tbl, "vision_range", config.vision_range);
    lua_get(tbl, "attack_range", config.attack_range);
    lua_get(tbl, "initial_energy", config.initial_energy);
    lua_get(tbl, "energy_drain_per_tick", config.energy_drain_per_tick);
    lua_get(tbl, "energy_gain_from_kill", config.energy_gain_from_kill);
    lua_get(tbl, "energy_gain_from_food", config.energy_gain_from_food);
    lua_get(tbl, "food_pickup_range", config.food_pickup_range);
    lua_get(tbl, "food_count", config.food_count);
    lua_get(tbl, "food_respawn_rate", config.food_respawn_rate);
    lua_get(tbl, "mutation_rate", config.mutation_rate);
    lua_get(tbl, "crossover_rate", config.crossover_rate);
    lua_get(tbl, "weight_mutation_power", config.weight_mutation_power);
    lua_get(tbl, "add_node_rate", config.add_node_rate);
    lua_get(tbl, "add_connection_rate", config.add_connection_rate);
    lua_get(tbl, "delete_connection_rate", config.delete_connection_rate);
    lua_get(tbl, "max_hidden_nodes", config.max_hidden_nodes);
    lua_get(tbl, "generation_ticks", config.generation_ticks);
    lua_get(tbl, "max_generations", config.max_generations);
    lua_get(tbl, "compatibility_threshold", config.compatibility_threshold);
    lua_get(tbl, "c1_excess", config.c1_excess);
    lua_get(tbl, "c2_disjoint", config.c2_disjoint);
    lua_get(tbl, "c3_weight", config.c3_weight);
    lua_get(tbl, "stagnation_limit", config.stagnation_limit);
    lua_get(tbl, "target_fps", config.target_fps);
    lua_get_uint64(tbl, "seed", config.seed);
    lua_get(tbl, "output_dir", config.output_dir);
    lua_get(tbl, "log_interval", config.log_interval);
    lua_get(tbl, "fitness_survival_weight", config.fitness_survival_weight);
    lua_get(tbl, "fitness_kill_weight", config.fitness_kill_weight);
    lua_get(tbl, "fitness_energy_weight", config.fitness_energy_weight);
    lua_get(tbl, "fitness_distance_weight", config.fitness_distance_weight);
    lua_get(tbl, "complexity_penalty_weight", config.complexity_penalty_weight);
    lua_get(tbl, "activation_function", config.activation_function);
    lua_get_bool(tbl, "tick_log_enabled", config.tick_log_enabled);
    lua_get(tbl, "tick_log_interval", config.tick_log_interval);

    return config;
}

} // anonymous namespace

// ── Lua config loading ──────────────────────────────────────────────────

std::map<std::string, SimulationConfig> load_all_configs_lua(const std::string& filepath) {
    std::map<std::string, SimulationConfig> configs;

    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::table, sol::lib::string);

    // Inject C++ defaults as a Lua global so configs can reference them
    // without depending on a default.lua file.  Lua configs use:
    //   local base = moonai_defaults
    {
        SimulationConfig d;
        sol::table t = lua.create_table();
        t["grid_width"]                = d.grid_width;
        t["grid_height"]               = d.grid_height;
        t["boundary_mode"]             = (d.boundary_mode == BoundaryMode::Wrap) ? "wrap" : "clamp";
        t["predator_count"]            = d.predator_count;
        t["prey_count"]                = d.prey_count;
        t["predator_speed"]            = d.predator_speed;
        t["prey_speed"]                = d.prey_speed;
        t["vision_range"]              = d.vision_range;
        t["attack_range"]              = d.attack_range;
        t["initial_energy"]            = d.initial_energy;
        t["energy_drain_per_tick"]     = d.energy_drain_per_tick;
        t["energy_gain_from_kill"]     = d.energy_gain_from_kill;
        t["energy_gain_from_food"]     = d.energy_gain_from_food;
        t["food_pickup_range"]         = d.food_pickup_range;
        t["food_count"]                = d.food_count;
        t["food_respawn_rate"]         = d.food_respawn_rate;
        t["mutation_rate"]             = d.mutation_rate;
        t["crossover_rate"]            = d.crossover_rate;
        t["weight_mutation_power"]     = d.weight_mutation_power;
        t["add_node_rate"]             = d.add_node_rate;
        t["add_connection_rate"]       = d.add_connection_rate;
        t["delete_connection_rate"]    = d.delete_connection_rate;
        t["max_hidden_nodes"]          = d.max_hidden_nodes;
        t["generation_ticks"]          = d.generation_ticks;
        t["max_generations"]           = d.max_generations;
        t["compatibility_threshold"]   = d.compatibility_threshold;
        t["c1_excess"]                 = d.c1_excess;
        t["c2_disjoint"]               = d.c2_disjoint;
        t["c3_weight"]                 = d.c3_weight;
        t["stagnation_limit"]          = d.stagnation_limit;
        t["target_fps"]                = d.target_fps;
        t["seed"]                      = static_cast<double>(d.seed);
        t["output_dir"]                = d.output_dir;
        t["log_interval"]              = d.log_interval;
        t["fitness_survival_weight"]   = d.fitness_survival_weight;
        t["fitness_kill_weight"]       = d.fitness_kill_weight;
        t["fitness_energy_weight"]     = d.fitness_energy_weight;
        t["fitness_distance_weight"]   = d.fitness_distance_weight;
        t["complexity_penalty_weight"] = d.complexity_penalty_weight;
        t["activation_function"]       = d.activation_function;
        t["tick_log_enabled"]          = d.tick_log_enabled;
        t["tick_log_interval"]         = d.tick_log_interval;
        lua["moonai_defaults"] = t;
    }

    try {
        sol::protected_function_result result = lua.safe_script_file(filepath);
        if (!result.valid()) {
            sol::error err = result;
            spdlog::error("Lua config error in '{}': {}", filepath, err.what());
            return configs;
        }

        sol::object obj = result;
        if (obj.get_type() != sol::type::table) {
            spdlog::error("Lua config '{}' must return a table", filepath);
            return configs;
        }

        sol::table tbl = obj.as<sol::table>();

        for (auto& [key, val] : tbl) {
            if (key.get_type() == sol::type::string && val.get_type() == sol::type::table) {
                std::string name = key.as<std::string>();
                configs[name] = table_to_config(val.as<sol::table>());
            }
        }

        if (configs.empty()) {
            spdlog::error("Lua config '{}' returned no named experiments. "
                          "Expected: return {{ name = {{ ...params... }}, ... }}", filepath);
        } else {
            spdlog::info("Loaded {} experiment(s) from '{}'.", configs.size(), filepath);
        }
        return configs;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load Lua config '{}': {}", filepath, e.what());
        return configs;
    }
}

// ── JSON output (for config snapshots) ──────────────────────────────────

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

// ── Validation ──────────────────────────────────────────────────────────

std::vector<ConfigError> validate_config(const SimulationConfig& config) {
    std::vector<ConfigError> errors;

    auto check = [&](bool condition, const char* field, const char* msg) {
        if (!condition) {
            errors.push_back({field, msg});
        }
    };

    // Environment
    check(config.grid_width >= 100, "grid_width", "must be >= 100");
    check(config.grid_width <= 20000, "grid_width", "must be <= 20000");
    check(config.grid_height >= 100, "grid_height", "must be >= 100");
    check(config.grid_height <= 20000, "grid_height", "must be <= 20000");

    // Population
    check(config.predator_count >= 1, "predator_count", "must be >= 1");
    check(config.prey_count >= 1, "prey_count", "must be >= 1");
    check(config.predator_count + config.prey_count <= 50000,
          "population", "total population must be <= 50000");

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

// ── CLI override ────────────────────────────────────────────────────────

std::vector<ConfigError> apply_overrides(
    SimulationConfig& config,
    const std::vector<std::pair<std::string, std::string>>& overrides)
{
    std::vector<ConfigError> errors;

    for (const auto& [key, val] : overrides) {
        try {
            // Integer fields
            if (key == "grid_width") config.grid_width = std::stoi(val);
            else if (key == "grid_height") config.grid_height = std::stoi(val);
            else if (key == "predator_count") config.predator_count = std::stoi(val);
            else if (key == "prey_count") config.prey_count = std::stoi(val);
            else if (key == "food_count") config.food_count = std::stoi(val);
            else if (key == "max_hidden_nodes") config.max_hidden_nodes = std::stoi(val);
            else if (key == "generation_ticks") config.generation_ticks = std::stoi(val);
            else if (key == "max_generations") config.max_generations = std::stoi(val);
            else if (key == "stagnation_limit") config.stagnation_limit = std::stoi(val);
            else if (key == "target_fps") config.target_fps = std::stoi(val);
            else if (key == "log_interval") config.log_interval = std::stoi(val);
            else if (key == "tick_log_interval") config.tick_log_interval = std::stoi(val);
            // uint64 fields
            else if (key == "seed") config.seed = std::stoull(val);
            // Float fields
            else if (key == "predator_speed") config.predator_speed = std::stof(val);
            else if (key == "prey_speed") config.prey_speed = std::stof(val);
            else if (key == "vision_range") config.vision_range = std::stof(val);
            else if (key == "attack_range") config.attack_range = std::stof(val);
            else if (key == "initial_energy") config.initial_energy = std::stof(val);
            else if (key == "energy_drain_per_tick") config.energy_drain_per_tick = std::stof(val);
            else if (key == "energy_gain_from_kill") config.energy_gain_from_kill = std::stof(val);
            else if (key == "energy_gain_from_food") config.energy_gain_from_food = std::stof(val);
            else if (key == "food_pickup_range") config.food_pickup_range = std::stof(val);
            else if (key == "food_respawn_rate") config.food_respawn_rate = std::stof(val);
            else if (key == "mutation_rate") config.mutation_rate = std::stof(val);
            else if (key == "crossover_rate") config.crossover_rate = std::stof(val);
            else if (key == "weight_mutation_power") config.weight_mutation_power = std::stof(val);
            else if (key == "add_node_rate") config.add_node_rate = std::stof(val);
            else if (key == "add_connection_rate") config.add_connection_rate = std::stof(val);
            else if (key == "delete_connection_rate") config.delete_connection_rate = std::stof(val);
            else if (key == "compatibility_threshold") config.compatibility_threshold = std::stof(val);
            else if (key == "c1_excess") config.c1_excess = std::stof(val);
            else if (key == "c2_disjoint") config.c2_disjoint = std::stof(val);
            else if (key == "c3_weight") config.c3_weight = std::stof(val);
            else if (key == "fitness_survival_weight") config.fitness_survival_weight = std::stof(val);
            else if (key == "fitness_kill_weight") config.fitness_kill_weight = std::stof(val);
            else if (key == "fitness_energy_weight") config.fitness_energy_weight = std::stof(val);
            else if (key == "fitness_distance_weight") config.fitness_distance_weight = std::stof(val);
            else if (key == "complexity_penalty_weight") config.complexity_penalty_weight = std::stof(val);
            // String fields
            else if (key == "boundary_mode") {
                if (val == "wrap") config.boundary_mode = BoundaryMode::Wrap;
                else if (val == "clamp") config.boundary_mode = BoundaryMode::Clamp;
                else errors.push_back({key, "must be 'wrap' or 'clamp'"});
            }
            else if (key == "output_dir") config.output_dir = val;
            else if (key == "activation_function") config.activation_function = val;
            // Bool fields
            else if (key == "tick_log_enabled") config.tick_log_enabled = (val == "true" || val == "1");
            // Unknown key
            else {
                errors.push_back({key, "unknown config key"});
            }
        } catch (const std::exception& e) {
            errors.push_back({key, std::string("invalid value '") + val + "': " + e.what()});
        }
    }

    return errors;
}

// ── Float overrides (from Lua hooks) ─────────────────────────────────────

void apply_overrides_float(SimulationConfig& config,
                           const std::map<std::string, float>& overrides) {
    for (const auto& [key, val] : overrides) {
        // Float fields
        if      (key == "mutation_rate")             config.mutation_rate = val;
        else if (key == "crossover_rate")            config.crossover_rate = val;
        else if (key == "weight_mutation_power")     config.weight_mutation_power = val;
        else if (key == "add_node_rate")             config.add_node_rate = val;
        else if (key == "add_connection_rate")       config.add_connection_rate = val;
        else if (key == "delete_connection_rate")    config.delete_connection_rate = val;
        else if (key == "compatibility_threshold")   config.compatibility_threshold = val;
        else if (key == "c1_excess")                 config.c1_excess = val;
        else if (key == "c2_disjoint")               config.c2_disjoint = val;
        else if (key == "c3_weight")                 config.c3_weight = val;
        else if (key == "predator_speed")            config.predator_speed = val;
        else if (key == "prey_speed")                config.prey_speed = val;
        else if (key == "vision_range")              config.vision_range = val;
        else if (key == "attack_range")              config.attack_range = val;
        else if (key == "initial_energy")            config.initial_energy = val;
        else if (key == "energy_drain_per_tick")     config.energy_drain_per_tick = val;
        else if (key == "energy_gain_from_kill")     config.energy_gain_from_kill = val;
        else if (key == "energy_gain_from_food")     config.energy_gain_from_food = val;
        else if (key == "food_respawn_rate")         config.food_respawn_rate = val;
        else if (key == "fitness_survival_weight")   config.fitness_survival_weight = val;
        else if (key == "fitness_kill_weight")       config.fitness_kill_weight = val;
        else if (key == "fitness_energy_weight")     config.fitness_energy_weight = val;
        else if (key == "fitness_distance_weight")   config.fitness_distance_weight = val;
        else if (key == "complexity_penalty_weight") config.complexity_penalty_weight = val;
        // Integer fields (truncated from float)
        else if (key == "generation_ticks")          config.generation_ticks = static_cast<int>(val);
        else if (key == "stagnation_limit")          config.stagnation_limit = static_cast<int>(val);
        else if (key == "max_hidden_nodes")          config.max_hidden_nodes = static_cast<int>(val);
        else {
            spdlog::warn("Lua hook returned unknown override key: {}", key);
        }
    }
}

// ── CLI parsing ─────────────────────────────────────────────────────────

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
        }
        // New flags
        else if (arg == "--experiment" && i + 1 < argc) {
            args.experiment_name = argv[++i];
        } else if (arg == "--all") {
            args.run_all = true;
        } else if (arg == "--list") {
            args.list_experiments = true;
        } else if (arg == "--name" && i + 1 < argc) {
            args.run_name = argv[++i];
        } else if (arg == "--validate") {
            args.validate_only = true;
        } else if (arg == "--set" && i + 1 < argc) {
            std::string kv = argv[++i];
            auto eq = kv.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "Invalid --set format '%s' (expected key=value)\n", kv.c_str());
                args.help = true;
            } else {
                args.overrides.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
            }
        }
        // Positional argument: config path
        else if (arg[0] != '-') {
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
        "Usage: {} [OPTIONS] [config.lua]\n"
        "\n"
        "Options:\n"
        "  -c, --config <path>       Path to Lua config (default: config/default.lua)\n"
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
        "\n"
        "Experiment orchestration:\n"
        "      --experiment <name>   Select one experiment from a multi-config Lua file\n"
        "      --all                 Run all experiments sequentially (headless only)\n"
        "      --list                List experiment names and exit\n"
        "      --name <name>         Override output directory name\n"
        "      --set key=value       Override a config parameter (repeatable)\n"
        "      --validate            Validate config and exit\n"
        "\n"
        "  -h, --help                Show this help message\n",
        program_name
    );
}

} // namespace moonai
