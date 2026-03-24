#include <gtest/gtest.h>

#include "core/profiler_suite.hpp"

#include <filesystem>
#include <fstream>

using namespace moonai;

TEST(ProfilerSuiteConfigTest, LoadsProfilerSuitesFromLua) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / std::filesystem::path("moonai_profiler_suite_test.lua");
    std::ofstream handle(path);
    ASSERT_TRUE(handle.is_open());
    handle << "return {\n"
              "  baseline = {\n"
              "    config_path = 'config.lua',\n"
              "    experiment = 'baseline_seed42',\n"
              "    generations = 24,\n"
              "    output_dir = 'output/profiles',\n"
              "    seeds = { 41, 42, 43, 44, 45, 46 },\n"
              "  },\n"
              "}\n";
    handle.close();

    const auto suites = load_profiler_suites_lua(path.string());
    ASSERT_EQ(suites.size(), 1u);
    const auto it = suites.find("baseline");
    ASSERT_NE(it, suites.end());
    EXPECT_EQ(it->second.config_path, "config.lua");
    EXPECT_EQ(it->second.experiment_name, "baseline_seed42");
    EXPECT_EQ(it->second.generations, 24);
    EXPECT_EQ(it->second.output_dir, "output/profiles");
    ASSERT_EQ(it->second.seeds.size(), 6u);
    EXPECT_EQ(it->second.seeds.front(), 41u);
    EXPECT_EQ(it->second.seeds.back(), 46u);

    std::filesystem::remove(path);
}
