#pragma once

#include "core/config.hpp"
#include "core/random.hpp"
#include "core/types.hpp"
#include "data/logger.hpp"
#include "data/metrics.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/registry.hpp"
#include "simulation/simulation_manager.hpp"
#include "visualization/visualization_manager.hpp"

#include <csignal>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>

namespace moonai {

struct SessionConfig {
  SimulationConfig sim_config;
  std::string experiment_name;
  bool headless = false;
  bool interactive = true;
  bool enable_gpu = true;
  int speed_multiplier = 1;
  std::optional<std::string> run_name_override;
};

class Session {
public:
  explicit Session(const SessionConfig &cfg);

  bool run();

private:
  SessionConfig cfg_;
  Random rng_;
  Registry registry_;
  SimulationManager simulation_;
  EvolutionManager evolution_;
  MetricsCollector metrics_;
  Logger logger_;
  std::unique_ptr<VisualizationManager> visualization_;

  int steps_executed_ = 0;

  // Signal handling (static - shared across all sessions)
  static volatile std::sig_atomic_t g_running_;
  static void signal_handler(int);
  static void register_signal_handlers();

  // Internal helpers
  void step();
  StepMetrics record_and_log();
  void update_selected_visualization();
  bool should_continue() const;
  void log_report(const StepMetrics &snapshot) const;
  void log_early_stop(bool user_quit) const;
};

} // namespace moonai
