#pragma once

#include "core/config.hpp"

#include <string>
#include <fstream>
#include <cstdint>
#include <memory>
#include <vector>

namespace moonai {

class Agent;
class Genome;
class Species;
struct SimEvent;

class Logger {
public:
    Logger(const std::string& output_dir, std::uint64_t seed,
           const std::string& name = "");
    ~Logger();

    bool initialize(const SimulationConfig& config);
    void log_generation(int generation, int predator_count, int prey_count,
                        float best_fitness, float avg_fitness, int num_species,
                        float avg_complexity);
    void log_best_genome(int generation, const Genome& genome);
    void log_species(int generation, const std::vector<Species>& species);
    void log_tick(int generation, int tick,
                  const std::vector<std::unique_ptr<Agent>>& agents);
    void log_events(int generation, int tick,
                    const std::vector<SimEvent>& events);
    void flush_ticks();
    void flush();

    const std::string& run_dir() const { return run_dir_; }

private:
    std::string base_dir_;
    std::string run_dir_;
    std::string name_;
    std::uint64_t seed_;
    std::ofstream stats_file_;
    std::ofstream genomes_file_;
    std::ofstream species_file_;
    std::ofstream ticks_file_;
    std::string   ticks_buffer_;
    int           ticks_buffered_ = 0;
    std::ofstream events_file_;
    std::string   events_buffer_;
    int           events_buffered_ = 0;
    bool genomes_first_entry_ = true;

    static constexpr int TICK_FLUSH_EVERY = 500;
};

} // namespace moonai
