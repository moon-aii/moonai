#include "data/logger.hpp"
#include "data/metrics.hpp"
#include "evolution/genome.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace moonai;

class LoggerTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir_ = "/tmp/moonai_test_logs";
    std::filesystem::remove_all(test_dir_);
  }

  void TearDown() override {
    std::filesystem::remove_all(test_dir_);
  }

  std::string test_dir_;
};

TEST_F(LoggerTest, InitializeCreatesDirectory) {
  Logger logger(test_dir_, 42);
  SimulationConfig config;
  ASSERT_TRUE(logger.initialize(config));

  EXPECT_TRUE(std::filesystem::exists(logger.run_dir()));
  EXPECT_TRUE(std::filesystem::exists(logger.run_dir() + "/stats.csv"));
  EXPECT_TRUE(std::filesystem::exists(logger.run_dir() + "/genomes.json"));
  EXPECT_TRUE(std::filesystem::exists(logger.run_dir() + "/config.json"));
}

TEST_F(LoggerTest, StatsCSVHasSeedAndHeader) {
  Logger logger(test_dir_, 12345);
  SimulationConfig config;
  logger.initialize(config);
  StepMetrics metrics;
  metrics.step = 0;
  metrics.predator_count = 50;
  metrics.prey_count = 150;
  metrics.best_fitness = 1.5f;
  metrics.avg_fitness = 0.8f;
  metrics.num_species = 3;
  metrics.avg_genome_complexity = 5.2f;
  logger.log_report(metrics);
  logger.flush();

  std::ifstream f(logger.run_dir() + "/stats.csv");
  std::string line;

  std::getline(f, line);
  EXPECT_NE(line.find("12345"), std::string::npos);

  std::getline(f, line);
  EXPECT_NE(line.find("step"), std::string::npos);
  EXPECT_NE(line.find("best_fitness"), std::string::npos);

  std::getline(f, line);
  EXPECT_NE(line.find("0,50,150"), std::string::npos);
}

TEST_F(LoggerTest, GenomeExportIsValidJSON) {
  {
    Logger logger(test_dir_, 42);
    SimulationConfig config;
    logger.initialize(config);

    Genome g(2, 1);
    g.add_connection({0, 3, 0.5f, true, 0});
    g.set_fitness(1.23f);
    logger.log_best_genome(0, g);
    logger.flush();
  }

  std::ifstream f(
      std::filesystem::directory_iterator(test_dir_)->path().string() +
      "/genomes.json");
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

  auto j = nlohmann::json::parse(content);
  EXPECT_TRUE(j.is_array());
  EXPECT_EQ(j.size(), 1u);
  EXPECT_FLOAT_EQ(j[0]["fitness"].get<float>(), 1.23f);
  EXPECT_EQ(j[0]["step"].get<int>(), 0);
  EXPECT_TRUE(j[0].contains("genome"));
}

TEST_F(LoggerTest, ConfigSnapshotIsSaved) {
  Logger logger(test_dir_, 42);
  SimulationConfig config;
  config.grid_size = 999;
  logger.initialize(config);

  std::ifstream f(logger.run_dir() + "/config.json");
  auto j = nlohmann::json::parse(f);
  EXPECT_EQ(j["grid_size"].get<int>(), 999);
}

TEST_F(LoggerTest, RunDirContainsSeed) {
  Logger logger(test_dir_, 42);
  SimulationConfig config;
  logger.initialize(config);

  EXPECT_NE(logger.run_dir().find("seed42"), std::string::npos);
}
