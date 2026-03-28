#include "core/config.hpp"
#include "core/lua_runtime.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace moonai;

TEST(ConfigTest, LoadLuaMultiConfig) {
  std::string path =
      (std::filesystem::temp_directory_path() / "moonai_test_multi.lua")
          .string();
  {
    std::ofstream f(path);
    f << "return {\n"
      << "  baseline = { grid_size = 800, prey_count = 150 },\n"
      << "  big = { grid_size = 1600, prey_count = 300 },\n"
      << "}\n";
  }

  auto configs = load_all_configs_lua(path);
  EXPECT_EQ(configs.size(), 2u);
  EXPECT_TRUE(configs.count("baseline"));
  EXPECT_TRUE(configs.count("big"));
  EXPECT_EQ(configs["baseline"].grid_size, 800);
  EXPECT_EQ(configs["big"].grid_size, 1600);
  EXPECT_EQ(configs["big"].prey_count, 300);

  std::filesystem::remove(path);
}

TEST(ConfigTest, LoadNonexistentLuaReturnsEmpty) {
  auto configs = load_all_configs_lua("nonexistent_file.lua");
  EXPECT_TRUE(configs.empty());
}

TEST(ConfigTest, LoadInvalidLuaReturnsEmpty) {
  std::string path =
      (std::filesystem::temp_directory_path() / "moonai_test_bad.lua").string();
  {
    std::ofstream f(path);
    f << "this is not valid lua %%%";
  }

  auto configs = load_all_configs_lua(path);
  EXPECT_TRUE(configs.empty());

  std::filesystem::remove(path);
}

TEST(ConfigValidation, ValidDefaultConfig) {
  SimulationConfig config;
  auto errors = validate_config(config);
  EXPECT_TRUE(errors.empty());
}

TEST(ConfigValidation, InvalidGridSize) {
  SimulationConfig config;
  config.grid_size = 10;
  auto errors = validate_config(config);
  EXPECT_FALSE(errors.empty());
  bool found = false;
  for (const auto &e : errors) {
    if (e.field == "grid_size")
      found = true;
  }
  EXPECT_TRUE(found);
}

TEST(ConfigValidation, InvalidMutationRate) {
  SimulationConfig config;
  config.mutation_rate = 1.5f;
  auto errors = validate_config(config);
  EXPECT_FALSE(errors.empty());
}

TEST(ConfigValidation, AttackRangeExceedsVision) {
  SimulationConfig config;
  config.attack_range = 200.0f;
  config.vision_range = 100.0f;
  auto errors = validate_config(config);
  bool found = false;
  for (const auto &e : errors) {
    if (e.field == "attack_range")
      found = true;
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
  const char *argv[] = {"moonai"};
  auto args = parse_args(1, argv);
  EXPECT_EQ(args.config_path, "config.lua");
  EXPECT_FALSE(args.headless);
  EXPECT_FALSE(args.verbose);
  EXPECT_FALSE(args.help);
  EXPECT_EQ(args.seed_override, 0u);
}

TEST(CLIArgsTest, PositionalConfigPath) {
  const char *argv[] = {"moonai", "my_config.lua"};
  auto args = parse_args(2, argv);
  EXPECT_EQ(args.config_path, "my_config.lua");
}

TEST(CLIArgsTest, AllFlags) {
  const char *argv[] = {"moonai", "--headless", "-v", "-s",      "12345",
                        "-n",     "100",        "-c", "test.lua"};
  auto args = parse_args(9, argv);
  EXPECT_TRUE(args.headless);
  EXPECT_TRUE(args.verbose);
  EXPECT_EQ(args.seed_override, 12345u);
  EXPECT_EQ(args.max_steps_override, 100);
  EXPECT_EQ(args.config_path, "test.lua");
}

TEST(CLIArgsTest, HelpFlag) {
  const char *argv[] = {"moonai", "--help"};
  auto args = parse_args(2, argv);
  EXPECT_TRUE(args.help);
}

TEST(CLIArgsTest, NewExperimentFlags) {
  const char *argv[] = {
      "moonai", "experiments.lua", "--experiment", "mut_low",
      "--name", "test_run",        "--validate",
  };
  auto args = parse_args(7, argv);
  EXPECT_EQ(args.config_path, "experiments.lua");
  EXPECT_EQ(args.experiment_name, "mut_low");
  EXPECT_EQ(args.run_name, "test_run");
  EXPECT_TRUE(args.validate_only);
}

TEST(CLIArgsTest, AllAndListFlags) {
  const char *argv[] = {
      "moonai",
      "--all",
      "--list",
  };
  auto args = parse_args(3, argv);
  EXPECT_TRUE(args.run_all);
  EXPECT_TRUE(args.list_experiments);
}

TEST(CLIArgsTest, SetOverrides) {
  const char *argv[] = {
      "moonai", "--set", "mutation_rate=0.1", "--set", "prey_count=75",
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
      {"seed", "42"},
  };

  auto errors = apply_overrides(config, overrides);
  EXPECT_TRUE(errors.empty());
  EXPECT_FLOAT_EQ(config.mutation_rate, 0.1f);
  EXPECT_EQ(config.prey_count, 75);
  EXPECT_EQ(config.activation_function, "tanh");
  EXPECT_EQ(config.seed, 42u);
}

TEST(ApplyOverrides, BoundaryModeIsUnknownKey) {
  SimulationConfig config;
  std::vector<std::pair<std::string, std::string>> overrides = {
      {"boundary_mode", "clamp"},
  };

  auto errors = apply_overrides(config, overrides);
  EXPECT_EQ(errors.size(), 1u);
  EXPECT_EQ(errors[0].field, "boundary_mode");
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

TEST(ApplyOverrides, TargetFpsIsUnknownKey) {
  SimulationConfig config;
  std::vector<std::pair<std::string, std::string>> overrides = {
      {"target_fps", "240"},
  };

  auto errors = apply_overrides(config, overrides);
  EXPECT_EQ(errors.size(), 1u);
  EXPECT_EQ(errors[0].field, "target_fps");
}

TEST(ConfigJson, DoesNotSerializeTargetFps) {
  SimulationConfig config;
  auto json = config_to_json(config);
  EXPECT_FALSE(json.contains("target_fps"));
}
