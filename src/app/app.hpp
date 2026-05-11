#pragma once

#include "core/app_state.hpp"
#include "core/config.hpp"
#include "evolution/evolution_manager.hpp"
#include "metrics/logger.hpp"
#include "visualization/visualization_manager.hpp"

#include <csignal>
#include <memory>

namespace moonai {

class App {
public:
  explicit App(AppConfig cfg);

  bool run();

private:
  AppConfig cfg_;
  AppState state_;
  EvolutionManager evolution_;
  Logger logger_;
  std::unique_ptr<VisualizationManager> visualization_;

  static volatile std::sig_atomic_t g_running_;
  static void signal_handler(int);
  static void register_signal_handlers();

  bool step();
  void record_and_log();
  bool should_continue() const;
};

} // namespace moonai
