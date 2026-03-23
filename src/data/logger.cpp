#include "data/logger.hpp"
#include "evolution/genome.hpp"
#include "evolution/species.hpp"
#include "simulation/agent.hpp"
#include "simulation/simulation_manager.hpp"

#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace moonai {

Logger::Logger(const std::string& output_dir, std::uint64_t seed,
               const std::string& name)
    : base_dir_(output_dir)
    , name_(name)
    , seed_(seed) {
}

Logger::~Logger() {
    flush_ticks();
    if (events_file_.is_open() && !events_buffer_.empty()) {
        events_file_ << events_buffer_;
    }
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
    if (ticks_file_.is_open()) {
        ticks_file_.close();
    }
    if (events_file_.is_open()) {
        events_file_.close();
    }
}

bool Logger::initialize(const SimulationConfig& config) {
    // Create timestamped run directory
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
        // Named experiment: use name as-is (e.g., "baseline_seed42")
        dir_name = name_;
    } else {
        // Anonymous run: output/YYYYMMDD_HHMMSS_seed42/
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_seed" << seed_;
        dir_name = oss.str();
    }
    run_dir_ = base_dir_ + "/" + dir_name;

    // Overwrite protection: append suffix if directory exists
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

    // Save config snapshot
    save_config(config, run_dir_ + "/config.json");

    // Open stats CSV
    std::string stats_path = run_dir_ + "/stats.csv";
    stats_file_.open(stats_path);
    if (!stats_file_.is_open()) {
        spdlog::error("Failed to open log file: {}", stats_path);
        return false;
    }

    stats_file_ << "# master_seed=" << seed_ << "\n";
    stats_file_ << "generation,predator_count,prey_count,"
                   "best_fitness,avg_fitness,num_species,avg_complexity\n";

    // Open species CSV
    std::string species_path = run_dir_ + "/species.csv";
    species_file_.open(species_path);
    if (species_file_.is_open()) {
        species_file_ << "# master_seed=" << seed_ << "\n";
        species_file_ << "generation,species_id,size,avg_fitness,"
                         "best_fitness,stagnation_count\n";
    }

    // Open genomes JSON (streaming array)
    std::string genomes_path = run_dir_ + "/genomes.json";
    genomes_file_.open(genomes_path);
    if (!genomes_file_.is_open()) {
        spdlog::error("Failed to open genomes file: {}", genomes_path);
        return false;
    }
    genomes_file_ << "[";

    // Open ticks CSV only when per-tick logging is enabled
    if (config.tick_log_enabled) {
        std::string ticks_path = run_dir_ + "/ticks.csv";
        ticks_file_.open(ticks_path);
        if (!ticks_file_.is_open()) {
            spdlog::error("Failed to open ticks file: {}", ticks_path);
            return false;
        }
        ticks_file_ << "generation,tick,agent_id,type,alive,x,y,energy,kills,food_eaten\n";

        // Also open events CSV for interaction logging
        std::string events_path = run_dir_ + "/events.csv";
        events_file_.open(events_path);
        if (!events_file_.is_open()) {
            spdlog::error("Failed to open events file: {}", events_path);
            return false;
        }
        events_file_ << "generation,tick,event_type,agent_id,target_id,x,y\n";
    }

    spdlog::info("Logger initialized, output: {}", run_dir_);
    return true;
}

void Logger::log_generation(int generation, int predator_count, int prey_count,
                            float best_fitness, float avg_fitness, int num_species,
                            float avg_complexity) {
    if (!stats_file_.is_open()) return;

    stats_file_ << generation << ","
                << predator_count << ","
                << prey_count << ","
                << best_fitness << ","
                << avg_fitness << ","
                << num_species << ","
                << avg_complexity << "\n";
}

void Logger::log_best_genome(int generation, const Genome& genome) {
    if (!genomes_file_.is_open()) return;

    nlohmann::json j;
    j["generation"] = generation;
    j["fitness"] = genome.fitness();
    j["num_nodes"] = genome.nodes().size();
    j["num_connections"] = genome.connections().size();

    nlohmann::json nodes_json = nlohmann::json::array();
    for (const auto& node : genome.nodes()) {
        nodes_json.push_back({
            {"id", node.id},
            {"type", static_cast<int>(node.type)}
        });
    }
    j["nodes"] = nodes_json;

    nlohmann::json conns_json = nlohmann::json::array();
    for (const auto& conn : genome.connections()) {
        conns_json.push_back({
            {"in", conn.in_node},
            {"out", conn.out_node},
            {"weight", conn.weight},
            {"enabled", conn.enabled},
            {"innovation", conn.innovation}
        });
    }
    j["connections"] = conns_json;

    if (!genomes_first_entry_) {
        genomes_file_ << ",";
    }
    genomes_file_ << "\n" << j.dump(2);
    genomes_first_entry_ = false;
}

void Logger::log_species(int generation, const std::vector<Species>& species) {
    if (!species_file_.is_open()) return;

    for (const auto& s : species) {
        species_file_ << generation << ","
                      << s.id() << ","
                      << s.members().size() << ","
                      << s.average_fitness() << ","
                      << s.best_fitness_ever() << ","
                      << s.generations_without_improvement() << "\n";
    }
}

void Logger::log_tick(int generation, int tick,
                      const std::vector<std::unique_ptr<Agent>>& agents) {
    if (!ticks_file_.is_open()) return;

    for (const auto& agent : agents) {
        ticks_buffer_ += std::to_string(generation) + ","
            + std::to_string(tick) + ","
            + std::to_string(agent->id()) + ","
            + (agent->type() == AgentType::Predator ? "predator" : "prey") + ","
            + (agent->alive() ? "1" : "0") + ","
            + std::to_string(agent->position().x) + ","
            + std::to_string(agent->position().y) + ","
            + std::to_string(agent->energy()) + ","
            + std::to_string(agent->kills()) + ","
            + std::to_string(agent->food_eaten()) + "\n";
        ++ticks_buffered_;
    }

    if (ticks_buffered_ >= TICK_FLUSH_EVERY) {
        flush_ticks();
    }
}

void Logger::log_events(int generation, int tick,
                         const std::vector<SimEvent>& events) {
    if (!events_file_.is_open() || events.empty()) return;

    for (const auto& e : events) {
        events_buffer_ += std::to_string(generation) + ","
            + std::to_string(tick) + ","
            + (e.type == SimEvent::Kill ? "kill" : "food") + ","
            + std::to_string(e.agent_id) + ","
            + std::to_string(e.target_id) + ","
            + std::to_string(e.position.x) + ","
            + std::to_string(e.position.y) + "\n";
        ++events_buffered_;
    }

    if (events_buffered_ >= TICK_FLUSH_EVERY) {
        events_file_ << events_buffer_;
        events_buffer_.clear();
        events_buffered_ = 0;
    }
}

void Logger::flush_ticks() {
    if (!ticks_file_.is_open() || ticks_buffer_.empty()) return;
    ticks_file_ << ticks_buffer_;
    ticks_buffer_.clear();
    ticks_buffered_ = 0;
}

void Logger::flush() {
    if (stats_file_.is_open()) stats_file_.flush();
    if (genomes_file_.is_open()) genomes_file_.flush();
    if (species_file_.is_open()) species_file_.flush();
    flush_ticks();
    if (events_file_.is_open() && !events_buffer_.empty()) {
        events_file_ << events_buffer_;
        events_buffer_.clear();
        events_buffered_ = 0;
    }
    if (ticks_file_.is_open()) ticks_file_.flush();
}

} // namespace moonai
