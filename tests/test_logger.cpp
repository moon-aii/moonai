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

TEST_F(LoggerTest, RunDirContainsSeed) {
  Logger logger(test_dir_, 42);
  SimulationConfig config;
  logger.initialize(config);

  EXPECT_NE(logger.run_dir().find("seed42"), std::string::npos);
}
