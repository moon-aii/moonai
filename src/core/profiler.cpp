#include "core/profiler.hpp"

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

namespace moonai {

namespace {

template<typename Enum>
constexpr std::size_t enum_index(Enum value) {
    return static_cast<std::size_t>(value);
}

} // namespace

Profiler& Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

void Profiler::set_enabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_relaxed);
}

void Profiler::start_run(const std::string& experiment_name,
                         const std::string& output_dir,
                         std::uint64_t seed,
                         int predator_count,
                         int prey_count,
                         int food_count,
                         int generation_ticks,
                         bool gpu_allowed) {
    experiment_name_ = experiment_name;
    output_dir_ = output_dir;
    seed_ = seed;
    predator_count_ = predator_count;
    prey_count_ = prey_count;
    food_count_ = food_count;
    generation_ticks_ = generation_ticks;
    gpu_allowed_ = gpu_allowed;
    generation_gpu_used_ = false;
    generation_records_.clear();

    for (auto& duration : current_durations_ns_) {
        duration.store(0, std::memory_order_relaxed);
    }
    for (auto& counter : current_counters_) {
        counter.store(0, std::memory_order_relaxed);
    }
}

void Profiler::start_generation(int generation) {
    if (!enabled()) {
        return;
    }
    (void)generation;
    generation_gpu_used_ = false;
    for (auto& duration : current_durations_ns_) {
        duration.store(0, std::memory_order_relaxed);
    }
    for (auto& counter : current_counters_) {
        counter.store(0, std::memory_order_relaxed);
    }
}

void Profiler::mark_gpu_used(bool used) {
    if (!enabled() || !used) {
        return;
    }
    generation_gpu_used_ = true;
}

void Profiler::add_duration(ProfileEvent event, std::int64_t nanoseconds) {
    if (!enabled()) {
        return;
    }
    current_durations_ns_[enum_index(event)].fetch_add(nanoseconds, std::memory_order_relaxed);
}

void Profiler::set_duration(ProfileEvent event, std::int64_t nanoseconds) {
    if (!enabled()) {
        return;
    }
    current_durations_ns_[enum_index(event)].store(nanoseconds, std::memory_order_relaxed);
}

void Profiler::increment(ProfileCounter counter, std::int64_t value) {
    if (!enabled()) {
        return;
    }
    current_counters_[enum_index(counter)].fetch_add(value, std::memory_order_relaxed);
}

void Profiler::finish_generation(const GenerationProfileMeta& meta) {
    if (!enabled()) {
        return;
    }

    GenerationRecord record;
    record.meta = meta;
    record.meta.gpu_used = generation_gpu_used_;

    for (std::size_t i = 0; i < record.durations_ns.size(); ++i) {
        record.durations_ns[i] = current_durations_ns_[i].load(std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < record.counters.size(); ++i) {
        record.counters[i] = current_counters_[i].load(std::memory_order_relaxed);
    }

    generation_records_.push_back(std::move(record));
}

void Profiler::finish_run(std::int64_t run_total_ns) {
    if (!enabled() || output_dir_.empty()) {
        return;
    }

    std::filesystem::create_directories(output_dir_);

    const auto csv_path = std::filesystem::path(output_dir_) / "profile_generations.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "generation,gpu_used,predator_count,prey_count,species_count,best_fitness,avg_fitness,avg_complexity";
        for (std::size_t i = 0; i < enum_index(ProfileEvent::Count); ++i) {
            csv << "," << profile_event_name(static_cast<ProfileEvent>(i)) << "_ms";
        }
        for (std::size_t i = 0; i < enum_index(ProfileCounter::Count); ++i) {
            csv << "," << profile_counter_name(static_cast<ProfileCounter>(i));
        }
        csv << "\n";

        for (const auto& record : generation_records_) {
            csv << record.meta.generation << ","
                << (record.meta.gpu_used ? 1 : 0) << ","
                << record.meta.predator_count << ","
                << record.meta.prey_count << ","
                << record.meta.species_count << ","
                << record.meta.best_fitness << ","
                << record.meta.avg_fitness << ","
                << record.meta.avg_complexity;
            for (const auto duration_ns : record.durations_ns) {
                csv << "," << (static_cast<double>(duration_ns) / 1'000'000.0);
            }
            for (const auto counter : record.counters) {
                csv << "," << counter;
            }
            csv << "\n";
        }
    }

    nlohmann::json summary;
    summary["experiment_name"] = experiment_name_;
    summary["seed"] = seed_;
    summary["predator_count"] = predator_count_;
    summary["prey_count"] = prey_count_;
    summary["food_count"] = food_count_;
    summary["generation_ticks"] = generation_ticks_;
    summary["gpu_allowed"] = gpu_allowed_;
    summary["generation_count"] = generation_records_.size();
    summary["run_total_ms"] = static_cast<double>(run_total_ns) / 1'000'000.0;

    std::array<std::int64_t, enum_index(ProfileEvent::Count)> total_durations{};
    std::array<std::int64_t, enum_index(ProfileCounter::Count)> total_counters{};
    int gpu_generations = 0;

    for (const auto& record : generation_records_) {
        if (record.meta.gpu_used) {
            ++gpu_generations;
        }
        for (std::size_t i = 0; i < total_durations.size(); ++i) {
            total_durations[i] += record.durations_ns[i];
        }
        for (std::size_t i = 0; i < total_counters.size(); ++i) {
            total_counters[i] += record.counters[i];
        }
    }

    summary["gpu_generation_count"] = gpu_generations;

    nlohmann::json durations_json;
    for (std::size_t i = 0; i < total_durations.size(); ++i) {
        durations_json[profile_event_name(static_cast<ProfileEvent>(i))] = {
            {"total_ms", static_cast<double>(total_durations[i]) / 1'000'000.0},
            {"avg_ms_per_generation", generation_records_.empty()
                ? 0.0
                : static_cast<double>(total_durations[i]) / 1'000'000.0 / static_cast<double>(generation_records_.size())}
        };
    }
    summary["durations"] = durations_json;

    nlohmann::json counters_json;
    for (std::size_t i = 0; i < total_counters.size(); ++i) {
        counters_json[profile_counter_name(static_cast<ProfileCounter>(i))] = total_counters[i];
    }
    summary["counters"] = counters_json;

    const auto json_path = std::filesystem::path(output_dir_) / "profile_summary.json";
    std::ofstream json(json_path);
    if (json.is_open()) {
        json << summary.dump(2) << "\n";
    }
}

const char* profile_event_name(ProfileEvent event) {
    switch (event) {
        case ProfileEvent::GenerationTotal: return "generation_total";
        case ProfileEvent::BuildNetworks: return "build_networks";
        case ProfileEvent::PrepareGpuGeneration: return "prepare_gpu_generation";
        case ProfileEvent::GpuSensorFlatten: return "gpu_sensor_flatten";
        case ProfileEvent::GpuPackInputs: return "gpu_pack_inputs";
        case ProfileEvent::GpuLaunch: return "gpu_launch";
        case ProfileEvent::GpuStartUnpack: return "gpu_start_unpack";
        case ProfileEvent::GpuFinishUnpack: return "gpu_finish_unpack";
        case ProfileEvent::GpuOutputConvert: return "gpu_output_convert";
        case ProfileEvent::CpuEvalTotal: return "cpu_eval_total";
        case ProfileEvent::CpuSensorBuild: return "cpu_sensor_build";
        case ProfileEvent::CpuNnActivate: return "cpu_nn_activate";
        case ProfileEvent::ApplyActions: return "apply_actions";
        case ProfileEvent::SimulationTick: return "simulation_tick";
        case ProfileEvent::RebuildSpatialGrid: return "rebuild_spatial_grid";
        case ProfileEvent::RebuildFoodGrid: return "rebuild_food_grid";
        case ProfileEvent::AgentUpdate: return "agent_update";
        case ProfileEvent::ProcessEnergy: return "process_energy";
        case ProfileEvent::ProcessFood: return "process_food";
        case ProfileEvent::ProcessAttacks: return "process_attacks";
        case ProfileEvent::BoundaryApply: return "boundary_apply";
        case ProfileEvent::DeathCheck: return "death_check";
        case ProfileEvent::CountAlive: return "count_alive";
        case ProfileEvent::ComputeFitness: return "compute_fitness";
        case ProfileEvent::Speciate: return "speciate";
        case ProfileEvent::RemoveStagnantSpecies: return "remove_stagnant_species";
        case ProfileEvent::Reproduce: return "reproduce";
        case ProfileEvent::Logging: return "logging";
        case ProfileEvent::TickCallback: return "tick_callback";
        case ProfileEvent::CompatibilityDistance: return "compatibility_distance";
        case ProfileEvent::PhysicsBuildSensors: return "physics_build_sensors";
        case ProfileEvent::SpatialQueryRadius: return "spatial_query_radius";
        case ProfileEvent::Count: return "count";
    }
    return "unknown";
}

const char* profile_counter_name(ProfileCounter counter) {
    switch (counter) {
        case ProfileCounter::TicksExecuted: return "ticks_executed";
        case ProfileCounter::GridQueryCalls: return "grid_query_calls";
        case ProfileCounter::GridCandidatesScanned: return "grid_candidates_scanned";
        case ProfileCounter::FoodEatAttempts: return "food_eat_attempts";
        case ProfileCounter::FoodEaten: return "food_eaten";
        case ProfileCounter::AttackChecks: return "attack_checks";
        case ProfileCounter::Kills: return "kills";
        case ProfileCounter::CompatibilityChecks: return "compatibility_checks";
        case ProfileCounter::Count: return "count";
    }
    return "unknown";
}

} // namespace moonai
