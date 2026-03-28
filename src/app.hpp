#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "data/logger.hpp"
#include "data/metrics.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/registry.hpp"
#include "simulation/simulation_manager.hpp"
#include "visualization/visualization_manager.hpp"

#include <csignal>
#include <memory>
#include <optional>
#include <unordered_map>

namespace moonai {

struct AppConfig {
  SimulationConfig sim_config;
  std::string experiment_name;
  bool headless = false;
  bool interactive = true;
  bool enable_gpu = true;
  int speed_multiplier = 1;
  std::optional<std::string> run_name_override;

  static constexpr bool cuda_compiled =
#ifdef MOONAI_ENABLE_CUDA
      true;
#else
      false;
#endif

  static constexpr bool openmp_compiled =
#ifdef MOONAI_OPENMP_ENABLED
      true;
#else
      false;
#endif

  static constexpr const char *platform =
#ifdef _WIN32
      "windows";
#else
      "linux";
#endif
};

class App {
public:
  explicit App(const AppConfig &cfg);

  bool run();

private:
  AppConfig cfg_;
  Random rng_;
  Registry registry_;
  SimulationManager simulation_;
  EvolutionManager evolution_;
  MetricsCollector metrics_;
  Logger logger_;
  std::unique_ptr<VisualizationManager> visualization_;

  struct RunEventTotals {
    int kills = 0;
    int food_eaten = 0;
    int births = 0;
    int deaths = 0;
  };

  int steps_executed_ = 0;
  std::vector<SimEvent> last_step_events_;
  RunEventTotals event_totals_;
  std::unordered_map<std::uint32_t, float> selected_node_activations_;

  static volatile std::sig_atomic_t g_running_;
  static void signal_handler(int);
  static void register_signal_handlers();

  void step();
  StepMetrics record_and_log();
  void update_selected_visualization();
  FrameSnapshot build_frame_snapshot() const;
  void accumulate_events(const std::vector<SimEvent> &events);
  bool should_continue() const;
  void log_report(const StepMetrics &snapshot) const;
  void log_early_stop(bool user_quit) const;
};

} // namespace moonai
