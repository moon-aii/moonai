#include <gtest/gtest.h>
#include "core/config.hpp"
#include "core/lua_runtime.hpp"

#include <fstream>
#include <filesystem>

using namespace moonai;

TEST(ConfigTest, DefaultValues) {
    SimulationConfig config;

    EXPECT_EQ(config.grid_width, 4300);
    EXPECT_EQ(config.grid_height, 2400);
    EXPECT_EQ(config.predator_count, 500);
    EXPECT_EQ(config.prey_count, 1500);
    EXPECT_GT(config.mutation_rate, 0.0f);
    EXPECT_EQ(config.boundary_mode, BoundaryMode::Wrap);
    EXPECT_EQ(config.max_generations, 0);
    EXPECT_FLOAT_EQ(config.initial_energy, 150.0f);
}

TEST(ConfigTest, LoadLuaSingleNamedConfig) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_test_config.lua").string();
    {
        std::ofstream f(path);
        f << "return { default = { grid_width = 1024, prey_count = 200, boundary_mode = 'clamp' } }";
    }

    auto configs = load_all_configs_lua(path);
    ASSERT_EQ(configs.size(), 1u);
    ASSERT_TRUE(configs.count("default"));
    const auto& config = configs["default"];
    EXPECT_EQ(config.grid_width, 1024);
    EXPECT_EQ(config.prey_count, 200);
    EXPECT_EQ(config.boundary_mode, BoundaryMode::Clamp);
    // Unspecified fields keep defaults
    EXPECT_EQ(config.predator_count, 500);

    std::filesystem::remove(path);
}

TEST(ConfigTest, LoadLuaMultiConfig) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_test_multi.lua").string();
    {
        std::ofstream f(path);
        f << "return {\n"
          << "  baseline = { grid_width = 800, prey_count = 150 },\n"
          << "  big = { grid_width = 1600, prey_count = 300 },\n"
          << "}\n";
    }

    auto configs = load_all_configs_lua(path);
    EXPECT_EQ(configs.size(), 2u);
    EXPECT_TRUE(configs.count("baseline"));
    EXPECT_TRUE(configs.count("big"));
    EXPECT_EQ(configs["baseline"].grid_width, 800);
    EXPECT_EQ(configs["big"].grid_width, 1600);
    EXPECT_EQ(configs["big"].prey_count, 300);

    std::filesystem::remove(path);
}

TEST(ConfigTest, LoadLuaNamedMapWithCustomName) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_test_named.lua").string();
    {
        std::ofstream f(path);
        f << "return { my_run = { grid_width = 999 } }";
    }

    auto configs = load_all_configs_lua(path);
    EXPECT_EQ(configs.size(), 1u);
    EXPECT_TRUE(configs.count("my_run"));
    EXPECT_EQ(configs["my_run"].grid_width, 999);
    // Unspecified fields keep defaults
    EXPECT_EQ(configs["my_run"].predator_count, 500);

    std::filesystem::remove(path);
}

TEST(ConfigTest, LoadNonexistentLuaReturnsEmpty) {
    auto configs = load_all_configs_lua("nonexistent_file.lua");
    EXPECT_TRUE(configs.empty());
}

TEST(ConfigTest, LoadInvalidLuaReturnsEmpty) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_test_bad.lua").string();
    {
        std::ofstream f(path);
        f << "this is not valid lua %%%";
    }

    auto configs = load_all_configs_lua(path);
    EXPECT_TRUE(configs.empty());

    std::filesystem::remove(path);
}

TEST(ConfigTest, SaveAndReloadJSON) {
    SimulationConfig original;
    original.grid_width = 1234;
    original.seed = 42;
    original.boundary_mode = BoundaryMode::Clamp;

    std::string path = (std::filesystem::temp_directory_path() / "moonai_test_save.json").string();
    save_config(original, path);

    // Verify the JSON is parseable and correct
    {
        std::ifstream f(path);
        auto j = nlohmann::json::parse(f);
        EXPECT_EQ(j["grid_width"].get<int>(), 1234);
        EXPECT_EQ(j["seed"].get<std::uint64_t>(), 42u);
        EXPECT_EQ(j["boundary_mode"].get<std::string>(), "clamp");
    }

    std::filesystem::remove(path);
}

TEST(ConfigValidation, ValidDefaultConfig) {
    SimulationConfig config;
    auto errors = validate_config(config);
    EXPECT_TRUE(errors.empty());
}

TEST(ConfigValidation, InvalidGridSize) {
    SimulationConfig config;
    config.grid_width = 10;  // too small
    auto errors = validate_config(config);
    EXPECT_FALSE(errors.empty());
    bool found = false;
    for (const auto& e : errors) {
        if (e.field == "grid_width") found = true;
    }
    EXPECT_TRUE(found);
}

TEST(ConfigValidation, InvalidMutationRate) {
    SimulationConfig config;
    config.mutation_rate = 1.5f;  // > 1
    auto errors = validate_config(config);
    EXPECT_FALSE(errors.empty());
}

TEST(ConfigValidation, AttackRangeExceedsVision) {
    SimulationConfig config;
    config.attack_range = 200.0f;
    config.vision_range = 100.0f;
    auto errors = validate_config(config);
    bool found = false;
    for (const auto& e : errors) {
        if (e.field == "attack_range") found = true;
    }
    EXPECT_TRUE(found);
}

TEST(ConfigValidation, ZeroPopulation) {
    SimulationConfig config;
    config.predator_count = 0;
    auto errors = validate_config(config);
    EXPECT_FALSE(errors.empty());
}

TEST(CLIArgsTest, DefaultArgs) {
    char* argv[] = {const_cast<char*>("moonai")};
    auto args = parse_args(1, argv);
    EXPECT_EQ(args.config_path, "config.lua");
    EXPECT_FALSE(args.headless);
    EXPECT_FALSE(args.verbose);
    EXPECT_FALSE(args.help);
    EXPECT_EQ(args.seed_override, 0u);
}

TEST(CLIArgsTest, PositionalConfigPath) {
    char* argv[] = {const_cast<char*>("moonai"), const_cast<char*>("my_config.lua")};
    auto args = parse_args(2, argv);
    EXPECT_EQ(args.config_path, "my_config.lua");
}

TEST(CLIArgsTest, AllFlags) {
    char* argv[] = {
        const_cast<char*>("moonai"),
        const_cast<char*>("--headless"),
        const_cast<char*>("-v"),
        const_cast<char*>("-s"), const_cast<char*>("12345"),
        const_cast<char*>("-g"), const_cast<char*>("100"),
        const_cast<char*>("-c"), const_cast<char*>("test.lua")
    };
    auto args = parse_args(9, argv);
    EXPECT_TRUE(args.headless);
    EXPECT_TRUE(args.verbose);
    EXPECT_EQ(args.seed_override, 12345u);
    EXPECT_EQ(args.max_generations_override, 100);
    EXPECT_EQ(args.config_path, "test.lua");
}

TEST(CLIArgsTest, HelpFlag) {
    char* argv[] = {const_cast<char*>("moonai"), const_cast<char*>("--help")};
    auto args = parse_args(2, argv);
    EXPECT_TRUE(args.help);
}

TEST(CLIArgsTest, NewExperimentFlags) {
    char* argv[] = {
        const_cast<char*>("moonai"),
        const_cast<char*>("experiments.lua"),
        const_cast<char*>("--experiment"), const_cast<char*>("mut_low"),
        const_cast<char*>("--name"), const_cast<char*>("test_run"),
        const_cast<char*>("--validate"),
    };
    auto args = parse_args(7, argv);
    EXPECT_EQ(args.config_path, "experiments.lua");
    EXPECT_EQ(args.experiment_name, "mut_low");
    EXPECT_EQ(args.run_name, "test_run");
    EXPECT_TRUE(args.validate_only);
}

TEST(CLIArgsTest, AllAndListFlags) {
    char* argv[] = {
        const_cast<char*>("moonai"),
        const_cast<char*>("--all"),
        const_cast<char*>("--list"),
    };
    auto args = parse_args(3, argv);
    EXPECT_TRUE(args.run_all);
    EXPECT_TRUE(args.list_experiments);
}

TEST(CLIArgsTest, SetOverrides) {
    char* argv[] = {
        const_cast<char*>("moonai"),
        const_cast<char*>("--set"), const_cast<char*>("mutation_rate=0.1"),
        const_cast<char*>("--set"), const_cast<char*>("prey_count=75"),
    };
    auto args = parse_args(5, argv);
    ASSERT_EQ(args.overrides.size(), 2u);
    EXPECT_EQ(args.overrides[0].first, "mutation_rate");
    EXPECT_EQ(args.overrides[0].second, "0.1");
    EXPECT_EQ(args.overrides[1].first, "prey_count");
    EXPECT_EQ(args.overrides[1].second, "75");
}

TEST(ApplyOverrides, ValidOverrides) {
    SimulationConfig config;
    std::vector<std::pair<std::string, std::string>> overrides = {
        {"mutation_rate", "0.1"},
        {"prey_count", "75"},
        {"activation_function", "tanh"},
        {"tick_log_enabled", "true"},
        {"boundary_mode", "clamp"},
        {"seed", "42"},
    };

    auto errors = apply_overrides(config, overrides);
    EXPECT_TRUE(errors.empty());
    EXPECT_FLOAT_EQ(config.mutation_rate, 0.1f);
    EXPECT_EQ(config.prey_count, 75);
    EXPECT_EQ(config.activation_function, "tanh");
    EXPECT_TRUE(config.tick_log_enabled);
    EXPECT_EQ(config.boundary_mode, BoundaryMode::Clamp);
    EXPECT_EQ(config.seed, 42u);
}

TEST(ApplyOverrides, UnknownKey) {
    SimulationConfig config;
    std::vector<std::pair<std::string, std::string>> overrides = {
        {"nonexistent_key", "value"},
    };

    auto errors = apply_overrides(config, overrides);
    EXPECT_EQ(errors.size(), 1u);
    EXPECT_EQ(errors[0].field, "nonexistent_key");
}

TEST(ApplyOverrides, InvalidValue) {
    SimulationConfig config;
    std::vector<std::pair<std::string, std::string>> overrides = {
        {"predator_count", "not_a_number"},
    };

    auto errors = apply_overrides(config, overrides);
    EXPECT_EQ(errors.size(), 1u);
    EXPECT_EQ(errors[0].field, "predator_count");
}

// ── LuaRuntime Tests ────────────────────────────────────────────────────

TEST(LuaRuntimeTest, LoadWithFitnessFn) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_rt_fitness.lua").string();
    {
        std::ofstream f(path);
        f << "return { test = {\n"
          << "  fitness_fn = function(s, w) return s.age_ratio * 10 end\n"
          << "} }\n";
    }

    LuaRuntime rt;
    auto configs = rt.load_config(path);
    ASSERT_EQ(configs.size(), 1u);
    rt.select_experiment("test");
    EXPECT_TRUE(rt.callbacks().has_fitness_fn);
    EXPECT_FALSE(rt.callbacks().has_on_generation_end);

    std::filesystem::remove(path);
}

TEST(LuaRuntimeTest, LoadWithoutFitnessFn) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_rt_nofn.lua").string();
    {
        std::ofstream f(path);
        f << "return { test = { grid_width = 800 } }\n";
    }

    LuaRuntime rt;
    auto configs = rt.load_config(path);
    ASSERT_EQ(configs.size(), 1u);
    rt.select_experiment("test");
    EXPECT_FALSE(rt.callbacks().has_fitness_fn);

    std::filesystem::remove(path);
}

TEST(LuaRuntimeTest, CallFitness) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_rt_call.lua").string();
    {
        std::ofstream f(path);
        f << "return { test = {\n"
          << "  fitness_fn = function(s, w) return s.age_ratio * 10 end\n"
          << "} }\n";
    }

    LuaRuntime rt;
    rt.load_config(path);
    rt.select_experiment("test");

    SimulationConfig config;
    float result = rt.call_fitness(0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, config);
    EXPECT_FLOAT_EQ(result, 5.0f);

    std::filesystem::remove(path);
}

TEST(LuaRuntimeTest, FitnessFnErrorReturnsFallback) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_rt_err.lua").string();
    {
        std::ofstream f(path);
        f << "return { test = {\n"
          << "  fitness_fn = function(s, w) error('broken') end\n"
          << "} }\n";
    }

    LuaRuntime rt;
    rt.load_config(path);
    rt.select_experiment("test");

    SimulationConfig config;
    float result = rt.call_fitness(0.5f, 1.0f, 0.5f, 1.0f, 0.2f, 0.1f, config);
    EXPECT_FLOAT_EQ(result, 0.0f);

    std::filesystem::remove(path);
}

TEST(LuaRuntimeTest, GenerationHookReturnsOverrides) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_rt_hook.lua").string();
    {
        std::ofstream f(path);
        f << "return { test = {\n"
          << "  on_generation_end = function(gen, stats)\n"
          << "    return { mutation_rate = 0.9 }\n"
          << "  end\n"
          << "} }\n";
    }

    LuaRuntime rt;
    rt.load_config(path);
    rt.select_experiment("test");
    EXPECT_TRUE(rt.callbacks().has_on_generation_end);

    GenerationStats gs{5, 1.0f, 0.5f, 3, 10, 20, 5.0f};
    std::map<std::string, float> overrides;
    bool has_overrides = rt.call_on_generation_end(gs, overrides);
    EXPECT_TRUE(has_overrides);
    ASSERT_TRUE(overrides.count("mutation_rate"));
    EXPECT_FLOAT_EQ(overrides["mutation_rate"], 0.9f);

    std::filesystem::remove(path);
}

TEST(LuaRuntimeTest, GenerationHookReturnsNil) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_rt_nil.lua").string();
    {
        std::ofstream f(path);
        f << "return { test = {\n"
          << "  on_generation_end = function(gen, stats)\n"
          << "    return nil\n"
          << "  end\n"
          << "} }\n";
    }

    LuaRuntime rt;
    rt.load_config(path);
    rt.select_experiment("test");

    GenerationStats gs{5, 1.0f, 0.5f, 3, 10, 20, 5.0f};
    std::map<std::string, float> overrides;
    bool has_overrides = rt.call_on_generation_end(gs, overrides);
    EXPECT_FALSE(has_overrides);
    EXPECT_TRUE(overrides.empty());

    std::filesystem::remove(path);
}
