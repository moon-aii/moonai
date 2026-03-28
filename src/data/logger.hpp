#pragma once

#include "core/config.hpp"
#include "simulation/entity.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace moonai {

class Genome;
class Species;
struct StepMetrics;

class Logger {
public:
  Logger(const std::string &output_dir, std::uint64_t seed,
         const std::string &name = "");
  ~Logger();

  bool initialize(const SimulationConfig &config);
  void log_report(const StepMetrics &metrics);
  void log_best_genome(int step, const Genome &genome);
  void log_species(int step, const std::vector<Species> &species);

  void flush();

  const std::string &run_dir() const {
    return run_dir_;
  }

private:
  std::string base_dir_;
  std::string run_dir_;
  std::string name_;
  std::uint64_t seed_;
  std::ofstream stats_file_;
  std::ofstream genomes_file_;
  std::ofstream species_file_;
  bool genomes_first_entry_ = true;
};

} // namespace moonai
