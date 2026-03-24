#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace moonai {

enum class ProfileEvent : std::size_t {
    GenerationTotal = 0,
    BuildNetworks,
    PrepareGpuGeneration,
    GpuSensorFlatten,
    GpuPackInputs,
    GpuLaunch,
    GpuStartUnpack,
    GpuFinishUnpack,
    GpuOutputConvert,
    CpuEvalTotal,
    CpuSensorBuild,
    CpuNnActivate,
    ApplyActions,
    SimulationTick,
    RebuildSpatialGrid,
    RebuildFoodGrid,
    AgentUpdate,
    ProcessEnergy,
    ProcessFood,
    ProcessAttacks,
    BoundaryApply,
    DeathCheck,
    CountAlive,
    ComputeFitness,
    Speciate,
    RemoveStagnantSpecies,
    Reproduce,
    Logging,
    TickCallback,
    CompatibilityDistance,
    PhysicsBuildSensors,
    SpatialQueryRadius,
    Count
};

enum class ProfileCounter : std::size_t {
    TicksExecuted = 0,
    GridQueryCalls,
    GridCandidatesScanned,
    FoodEatAttempts,
    FoodEaten,
    AttackChecks,
    Kills,
    CompatibilityChecks,
    Count
};

struct GenerationProfileMeta {
    int generation = 0;
    int predator_count = 0;
    int prey_count = 0;
    int species_count = 0;
    float best_fitness = 0.0f;
    float avg_fitness = 0.0f;
    float avg_complexity = 0.0f;
};

class Profiler {
public:
    static Profiler& instance();

    void set_enabled(bool enabled);
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

    void start_run(const std::string& experiment_name,
                   const std::string& output_root_dir,
                   std::uint64_t seed,
                   int predator_count,
                   int prey_count,
                   int food_count,
                   int generation_ticks,
                   bool gpu_allowed,
                   bool cuda_compiled,
                   bool openmp_compiled);
    void start_generation(int generation);
    void mark_cpu_used(bool used);
    void mark_gpu_used(bool used);
    void add_duration(ProfileEvent event, std::int64_t nanoseconds);
    void set_duration(ProfileEvent event, std::int64_t nanoseconds);
    void increment(ProfileCounter counter, std::int64_t value = 1);
    void finish_generation(const GenerationProfileMeta& meta);
    void finish_run(std::int64_t run_total_ns);

    const std::string& output_dir() const { return output_dir_; }

private:
    struct GenerationRecord {
        GenerationProfileMeta meta;
        bool cpu_used = false;
        bool gpu_used = false;
        std::array<std::int64_t, static_cast<std::size_t>(ProfileEvent::Count)> durations_ns{};
        std::array<std::int64_t, static_cast<std::size_t>(ProfileCounter::Count)> counters{};
    };

    Profiler() = default;

    std::atomic<bool> enabled_{false};
    std::string experiment_name_;
    std::string output_dir_;
    std::string generated_at_utc_;
    std::uint64_t seed_ = 0;
    int predator_count_ = 0;
    int prey_count_ = 0;
    int food_count_ = 0;
    int generation_ticks_ = 0;
    bool gpu_allowed_ = false;
    bool cuda_compiled_ = false;
    bool openmp_compiled_ = false;
    bool generation_cpu_used_ = false;
    bool generation_gpu_used_ = false;
    bool generation_active_ = false;
    std::chrono::steady_clock::time_point generation_start_{};
    std::vector<GenerationRecord> generation_records_;
    std::array<std::atomic<std::int64_t>, static_cast<std::size_t>(ProfileEvent::Count)> current_durations_ns_{};
    std::array<std::atomic<std::int64_t>, static_cast<std::size_t>(ProfileCounter::Count)> current_counters_{};
};

class ScopedTimer {
public:
    explicit ScopedTimer(ProfileEvent event)
        : event_(event)
        , active_(Profiler::instance().enabled()) {
        if (active_) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    ~ScopedTimer() {
        if (!active_) {
            return;
        }
        const auto end = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        Profiler::instance().add_duration(event_, ns);
    }

private:
    ProfileEvent event_;
    bool active_ = false;
    std::chrono::steady_clock::time_point start_{};
};

const char* profile_event_name(ProfileEvent event);
const char* profile_counter_name(ProfileCounter counter);

} // namespace moonai
