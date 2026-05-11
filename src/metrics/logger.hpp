#pragma once

#include "core/config.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace moonai {

class Genome;
class Species;
struct MetricsSnapshot;

class Logger {
public:
  Logger(const std::string &output_dir, int seed, const std::string &name = "");
  ~Logger();

  bool initialize(const SimulationConfig &config);
  void log_report(const MetricsSnapshot &metrics);
  void log_best_genome(int step, const Genome &genome);
  void log_species(int step, const std::vector<Species> &species, const std::string &population_name);

  void flush();

  const std::string &run_dir() const {
    return run_dir_;
  }

private:
  std::string base_dir_;
  std::string run_dir_;
  std::string name_;
  int seed_;
  std::ofstream stats_file_;
  std::ofstream genomes_file_;
  std::ofstream species_file_;
  bool genomes_first_entry_ = true;
};

} // namespace moonai
