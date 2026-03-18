#include <gtest/gtest.h>
#include "core/config.hpp"

#include <fstream>
#include <filesystem>

using namespace moonai;

TEST(ConfigTest, DefaultValues) {
    SimulationConfig config;

    EXPECT_EQ(config.grid_width, 800);
    EXPECT_EQ(config.grid_height, 600);
    EXPECT_EQ(config.predator_count, 50);
    EXPECT_EQ(config.prey_count, 150);
    EXPECT_GT(config.mutation_rate, 0.0f);
    EXPECT_EQ(config.boundary_mode, BoundaryMode::Wrap);
    EXPECT_EQ(config.max_generations, 0);
    EXPECT_FLOAT_EQ(config.initial_energy, 100.0f);
}

TEST(ConfigTest, LoadNonexistentFileReturnsDefaults) {
    auto config = load_config("nonexistent_file.json");

    EXPECT_EQ(config.grid_width, 800);
    EXPECT_EQ(config.predator_count, 50);
}

TEST(ConfigTest, LoadValidConfig) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_test_config.json").string();
    {
        std::ofstream f(path);
        f << R"({"grid_width": 1024, "prey_count": 200, "boundary_mode": "clamp"})";
    }

    auto config = load_config(path);
    EXPECT_EQ(config.grid_width, 1024);
    EXPECT_EQ(config.prey_count, 200);
    EXPECT_EQ(config.boundary_mode, BoundaryMode::Clamp);
    // Unspecified fields keep defaults
    EXPECT_EQ(config.predator_count, 50);

    std::filesystem::remove(path);
}

TEST(ConfigTest, LoadInvalidJsonReturnsDefaults) {
    std::string path = (std::filesystem::temp_directory_path() / "moonai_test_bad.json").string();
    {
        std::ofstream f(path);
        f << "not valid json {{{";
    }

    auto config = load_config(path);
    EXPECT_EQ(config.grid_width, 800);  // defaults

    std::filesystem::remove(path);
}

TEST(ConfigTest, SaveAndReload) {
    SimulationConfig original;
    original.grid_width = 1234;
    original.seed = 42;
    original.boundary_mode = BoundaryMode::Clamp;

    std::string path = (std::filesystem::temp_directory_path() / "moonai_test_save.json").string();
    save_config(original, path);

    auto loaded = load_config(path);
    EXPECT_EQ(loaded.grid_width, 1234);
    EXPECT_EQ(loaded.seed, 42u);
    EXPECT_EQ(loaded.boundary_mode, BoundaryMode::Clamp);

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
    EXPECT_EQ(args.config_path, "config/default_config.json");
    EXPECT_FALSE(args.headless);
    EXPECT_FALSE(args.verbose);
    EXPECT_FALSE(args.help);
    EXPECT_EQ(args.seed_override, 0u);
}

TEST(CLIArgsTest, PositionalConfigPath) {
    char* argv[] = {const_cast<char*>("moonai"), const_cast<char*>("my_config.json")};
    auto args = parse_args(2, argv);
    EXPECT_EQ(args.config_path, "my_config.json");
}

TEST(CLIArgsTest, AllFlags) {
    char* argv[] = {
        const_cast<char*>("moonai"),
        const_cast<char*>("--headless"),
        const_cast<char*>("-v"),
        const_cast<char*>("-s"), const_cast<char*>("12345"),
        const_cast<char*>("-g"), const_cast<char*>("100"),
        const_cast<char*>("-c"), const_cast<char*>("test.json")
    };
    auto args = parse_args(9, argv);
    EXPECT_TRUE(args.headless);
    EXPECT_TRUE(args.verbose);
    EXPECT_EQ(args.seed_override, 12345u);
    EXPECT_EQ(args.max_generations_override, 100);
    EXPECT_EQ(args.config_path, "test.json");
}

TEST(CLIArgsTest, HelpFlag) {
    char* argv[] = {const_cast<char*>("moonai"), const_cast<char*>("--help")};
    auto args = parse_args(2, argv);
    EXPECT_TRUE(args.help);
}
