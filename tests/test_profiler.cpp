#include <gtest/gtest.h>

#include "core/profiler.hpp"

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

using namespace moonai;

class ProfilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/moonai_test_profiles";
        std::filesystem::remove_all(test_dir_);
        Profiler::instance().set_enabled(true);
    }

    void TearDown() override {
        Profiler::instance().set_enabled(false);
        std::filesystem::remove_all(test_dir_);
    }

    std::string test_dir_;
};

TEST_F(ProfilerTest, WritesSingleJsonProfileWithSummaryAndGenerations) {
    auto& profiler = Profiler::instance();
    ProfileRunSpec spec;
    spec.experiment_name = "baseline_seed42";
    spec.output_root_dir = test_dir_;
    spec.seed = 42;
    spec.predator_count = 500;
    spec.prey_count = 1500;
    spec.food_count = 2500;
    spec.generation_ticks = 1500;
    spec.gpu_allowed = true;
    spec.cuda_compiled = true;
    spec.openmp_compiled = true;
    spec.suite_name = "baseline_suite";
    spec.base_experiment_name = "baseline_seed42";
    spec.config_fingerprint = "abc123";
    spec.profiler_entry_point = "moonai_profiler";
    profiler.start_run(spec);
    const std::string run_dir = profiler.output_dir();

    profiler.start_generation(0);
    profiler.mark_cpu_used(true);
    profiler.add_duration(ProfileEvent::BuildNetworks, 2'000'000);
    profiler.add_duration(ProfileEvent::CpuEvalTotal, 5'000'000);
    profiler.increment(ProfileCounter::TicksExecuted, 3);
    profiler.finish_generation({0, 490, 1400, 7, 12.5f, 4.0f, 18.0f});
    profiler.finish_run(8'000'000);

    ASSERT_TRUE(std::filesystem::exists(run_dir + "/profile.json"));

    std::ifstream handle(run_dir + "/profile.json");
    ASSERT_TRUE(handle.is_open());
    const auto profile = nlohmann::json::parse(handle);

    EXPECT_EQ(profile["schema_version"], 2);
    EXPECT_EQ(profile["run"]["experiment_name"], "baseline_seed42");
    EXPECT_EQ(profile["run"]["suite_name"], "baseline_suite");
    EXPECT_EQ(profile["run"]["config_fingerprint"], "abc123");
    EXPECT_EQ(profile["run"]["seed"], 42);
    EXPECT_EQ(profile["summary"]["generation_count"], 1);
    EXPECT_EQ(profile["summary"]["cpu_generation_count"], 1);
    EXPECT_EQ(profile["summary"]["gpu_generation_count"], 0);
    EXPECT_EQ(profile["generations"].size(), 1u);
    EXPECT_EQ(profile["generations"][0]["species_count"], 7);
    EXPECT_DOUBLE_EQ(
        profile["summary"]["events"]["cpu_eval_total"]["avg_ms_per_nonzero_generation"].get<double>(),
        5.0);
    EXPECT_EQ(profile["summary"]["counters"]["ticks_executed"]["total"], 3);
}

TEST_F(ProfilerTest, CreatesUniqueRunDirectories) {
    auto& profiler = Profiler::instance();
    ProfileRunSpec spec;
    spec.experiment_name = "baseline_seed42";
    spec.output_root_dir = test_dir_;
    spec.seed = 42;
    spec.predator_count = 500;
    spec.prey_count = 1500;
    spec.food_count = 2500;
    spec.generation_ticks = 1500;
    spec.gpu_allowed = true;
    spec.cuda_compiled = true;
    spec.openmp_compiled = true;
    profiler.start_run(spec);
    const std::string first_dir = profiler.output_dir();
    profiler.finish_run(0);

    profiler.start_run(spec);
    const std::string second_dir = profiler.output_dir();
    profiler.finish_run(0);

    EXPECT_NE(first_dir, second_dir);
    EXPECT_TRUE(std::filesystem::exists(first_dir + "/profile.json"));
    EXPECT_TRUE(std::filesystem::exists(second_dir + "/profile.json"));
}
