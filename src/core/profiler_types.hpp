#pragma once

#include <array>
#include <string_view>

namespace moonai {

struct ProfileEventDef {
  std::string_view name;
  bool is_window_duration;
  std::string_view description;
};

inline constexpr std::array<ProfileEventDef, 26> PROFILE_EVENTS = {{
    {"window_total", true,
     "Wall-clock report-window duration measured in headless mode."},
    {"prepare_gpu_window", false,
     "Time spent uploading GPU network state before a report window."},
    {"gpu_sensor_flatten", false,
     "Time spent preparing CPU-side sensor input buffers for the legacy GPU "
     "inference path."},
    {"gpu_pack_inputs", false,
     "Time spent packing flattened inputs into GPU buffers."},
    {"gpu_launch", false,
     "Time spent running GPU neural inference for the legacy hybrid path."},
    {"gpu_start_unpack", false, "Time spent starting GPU output unpacking."},
    {"gpu_finish_unpack", false,
     "Time spent waiting for GPU output unpacking to finish."},
    {"gpu_output_convert", false,
     "Time spent converting GPU outputs into agent actions."},
    {"cpu_eval_total", false,
     "Total CPU inference path time across a report window."},
    {"cpu_sensor_build", false, "Time spent building CPU-side sensor inputs."},
    {"cpu_nn_activate", false,
     "Time spent running CPU neural network activations."},
    {"apply_actions", false, "Time spent applying movement actions to agents."},
    {"simulation_step", false,
     "Accumulated time spent inside simulation steps."},
    {"rebuild_spatial_grid", false,
     "Time spent rebuilding the agent spatial grid."},
    {"rebuild_food_grid", false,
     "Time spent rebuilding the food spatial grid."},
    {"agent_update", false,
     "Time spent updating agent age and per-step state."},
    {"process_energy", false, "Time spent applying per-step energy drain."},
    {"process_food", false, "Time spent handling prey food pickup."},
    {"process_attacks", false, "Time spent handling predator attack checks."},
    {"boundary_apply", false, "Time spent applying world boundary rules."},
    {"death_check", false,
     "Time spent marking dead agents after step processing."},
    {"count_alive", false, "Time spent recounting living predators and prey."},
    {"compute_fitness", false, "Time spent computing genome fitness values."},
    {"speciate", false, "Time spent assigning genomes to species."},
    {"reproduce", false,
     "Time spent producing offspring during the report window."},
    {"logging", false, "Time spent writing report-window logs."},
}};

enum class ProfileEvent : std::size_t {
  WindowTotal = 0,
  PrepareGpuWindow = 1,
  GpuSensorFlatten = 2,
  GpuPackInputs = 3,
  GpuLaunch = 4,
  GpuStartUnpack = 5,
  GpuFinishUnpack = 6,
  GpuOutputConvert = 7,
  CpuEvalTotal = 8,
  CpuSensorBuild = 9,
  CpuNnActivate = 10,
  ApplyActions = 11,
  SimulationStep = 12,
  RebuildSpatialGrid = 13,
  RebuildFoodGrid = 14,
  AgentUpdate = 15,
  ProcessEnergy = 16,
  ProcessFood = 17,
  ProcessAttacks = 18,
  BoundaryApply = 19,
  DeathCheck = 20,
  CountAlive = 21,
  ComputeFitness = 22,
  Speciate = 23,
  Reproduce = 24,
  Logging = 25,
  Count = 26
};

constexpr std::size_t event_count = PROFILE_EVENTS.size();

} // namespace moonai

#ifndef MOONAI_BUILD_PROFILER

#define MOONAI_PROFILE_SCOPE(event) ((void)0)
#define MOONAI_PROFILE_MARK_CPU_USED(value) ((void)0)
#define MOONAI_PROFILE_MARK_GPU_USED(value) ((void)0)

#else

#include <chrono>

namespace moonai {
namespace profiler {

void mark_cpu_used(bool used);
void mark_gpu_used(bool used);

class ScopedTimer {
public:
  explicit ScopedTimer(ProfileEvent event);
  ~ScopedTimer();

private:
  ProfileEvent event_;
  bool active_;
  std::chrono::steady_clock::time_point start_;
};

} // namespace profiler
} // namespace moonai

#define MOONAI_PROFILE_SCOPE(event)                                            \
  ::moonai::profiler::ScopedTimer _moonai_scoped_timer(event)
#define MOONAI_PROFILE_MARK_CPU_USED(value)                                    \
  ::moonai::profiler::mark_cpu_used(value)
#define MOONAI_PROFILE_MARK_GPU_USED(value)                                    \
  ::moonai::profiler::mark_gpu_used(value)

#endif
