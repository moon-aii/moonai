#include "data/logger.hpp"

#include "core/app_state.hpp"
#include "evolution/genome.hpp"
#include "evolution/species.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sstream>

namespace moonai {

Logger::Logger(const std::string &output_dir, int seed, const std::string &name)
    : base_dir_(output_dir), name_(name), seed_(seed) {}

Logger::~Logger() {
  if (genomes_file_.is_open()) {
    genomes_file_ << "\n]";
    genomes_file_.close();
  }
  if (stats_file_.is_open()) {
    stats_file_.close();
  }
  if (species_file_.is_open()) {
    species_file_.close();
  }
}

bool Logger::initialize(const SimulationConfig &config) {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif

  std::string dir_name;
  if (!name_.empty()) {
    dir_name = name_;
  } else {
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_seed" << seed_;
    dir_name = oss.str();
  }
  run_dir_ = base_dir_ + "/" + dir_name;

  if (std::filesystem::exists(run_dir_)) {
    for (int suffix = 2; suffix < 1000; ++suffix) {
      std::string candidate = base_dir_ + "/" + dir_name + "_" + std::to_string(suffix);
      if (!std::filesystem::exists(candidate)) {
        run_dir_ = candidate;
        break;
      }
    }
  }

  std::filesystem::create_directories(run_dir_);
  save_config(config, run_dir_ + "/config.json");

  stats_file_.open(run_dir_ + "/stats.csv");
  if (!stats_file_.is_open()) {
    return false;
  }
  stats_file_ << "# master_seed=" << seed_ << "\n";
  stats_file_ << "step,predator_count,prey_count,births,deaths,"
                 "predator_species,prey_species,avg_complexity,avg_predator_energy,"
                 "avg_prey_energy\n";

  species_file_.open(run_dir_ + "/species.csv");
  if (species_file_.is_open()) {
    species_file_ << "# master_seed=" << seed_ << "\n";
    species_file_ << "step,population,species_id,size,avg_complexity\n";
  }

  genomes_file_.open(run_dir_ + "/genomes.json");
  if (!genomes_file_.is_open()) {
    return false;
  }
  genomes_file_ << "[";

  return true;
}

void Logger::log_report(const MetricsSnapshot &metrics) {
  if (!stats_file_.is_open()) {
    return;
  }
  stats_file_ << metrics.step << ',' << metrics.predator_count << ',' << metrics.prey_count << ',' << metrics.births
              << ',' << metrics.deaths << ',' << metrics.predator_species << ',' << metrics.prey_species << ','
              << metrics.avg_genome_complexity << ',' << metrics.avg_predator_energy << ',' << metrics.avg_prey_energy
              << '\n';
}

void Logger::log_best_genome(int step, const Genome &genome) {
  if (!genomes_file_.is_open()) {
    return;
  }

  nlohmann::json j;
  j["step"] = step;
  j["num_nodes"] = genome.nodes().size();
  j["num_connections"] = genome.connections().size();
  j["genome"] = nlohmann::json::parse(genome.to_json());

  if (!genomes_first_entry_) {
    genomes_file_ << ',';
  }
  genomes_file_ << '\n' << j.dump(2);
  genomes_first_entry_ = false;
}

void Logger::log_species(int step, const std::vector<Species> &species, const std::string &population_name) {
  if (!species_file_.is_open()) {
    return;
  }
  for (const auto &entry : species) {
    species_file_ << step << ',' << population_name << ',' << entry.id() << ',' << entry.members().size() << ','
                  << entry.average_complexity() << '\n';
  }
}

void Logger::flush() {
  if (stats_file_.is_open()) {
    stats_file_.flush();
  }
  if (genomes_file_.is_open()) {
    genomes_file_.flush();
  }
  if (species_file_.is_open()) {
    species_file_.flush();
  }
}

} // namespace moonai
