#include "app/app.hpp"

#include "core/profiler_macros.hpp"
#include "core/types.hpp"
#include "metrics/metrics.hpp"
#include "simulation/simulation.hpp"
#include "visualization/frame_snapshot.hpp"
#include "visualization/visualization_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace moonai {

volatile std::sig_atomic_t App::g_running_ = 1;

void App::signal_handler(int) {
  g_running_ = 0;
}

void App::register_signal_handlers() {
  static bool registered = false;
  if (!registered) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    registered = true;
  }
  g_running_ = 1;
}

App::App(AppConfig cfg)
    : cfg_([&cfg] {
        if (cfg.sim_config.seed == 0) {
          cfg.sim_config.seed = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        }
        return std::move(cfg);
      }()),
      state_(cfg_.sim_config.seed), evolution_(cfg_.sim_config),
      logger_(cfg_.sim_config.output_dir, cfg_.sim_config.seed, cfg_.run_name_override.value_or(cfg_.experiment_name)) {
  const auto errors = validate_config(cfg_.sim_config);
  if (!errors.empty()) {
    for (const auto &error : errors) {
      spdlog::error("Config error [{}]: {}", error.field, error.message);
    }
    throw std::runtime_error("Invalid simulation configuration");
  }

  state_.ui.speed_multiplier = cfg_.speed_multiplier;

  simulation::initialize(state_, cfg_.sim_config);
  evolution_.initialize(state_, SENSOR_COUNT, OUTPUT_COUNT);
  evolution_.seed_initial_population(state_);
  evolution_.refresh_species(state_);
  evolution_.initialize_inference(state_);
  metrics::refresh(state_);

  logger_.initialize(cfg_.sim_config);

  if (!cfg_.headless) {
    visualization_ = std::make_unique<VisualizationManager>(cfg_.sim_config, state_.ui);
    if (!visualization_->initialize()) {
      spdlog::error("Failed to initialize visualization");
      visualization_.reset();
    }
  }

  spdlog::info("CUDA initialized");

  register_signal_handlers();
}

bool App::step() {
  const auto run_step_once = [&]() {
    if (!simulation::prepare_step(state_, cfg_.sim_config)) {
      return false;
    }
    if (!evolution_.run_inference(state_)) {
      return false;
    }
    if (!simulation::resolve_step(state_, cfg_.sim_config)) {
      return false;
    }

    simulation::post_step(state_, cfg_.sim_config);
    evolution_.post_step(state_);
    return true;
  };

  if (!run_step_once()) {
    return false;
  }

  ++state_.runtime.step;
  metrics::refresh(state_);
  return true;
}

void App::record_and_log() {
  evolution_.refresh_species(state_);
  metrics::refresh(state_);

  logger_.log_report(state_.metrics);

  const Genome *best_genome = nullptr;
  std::size_t best_complexity = 0;
  for (const auto &genome : state_.predator.genomes) {
    const std::size_t complexity = genome.nodes().size() + genome.connections().size();
    if (!best_genome || complexity > best_complexity) {
      best_genome = &genome;
      best_complexity = complexity;
    }
  }
  for (const auto &genome : state_.prey.genomes) {
    const std::size_t complexity = genome.nodes().size() + genome.connections().size();
    if (!best_genome || complexity > best_complexity) {
      best_genome = &genome;
      best_complexity = complexity;
    }
  }

  if (best_genome) {
    logger_.log_best_genome(state_.metrics.step, *best_genome);
  }

  logger_.log_species(state_.metrics.step, state_.predator.species, "predator");
  logger_.log_species(state_.metrics.step, state_.prey.species, "prey");
  logger_.flush();

  spdlog::info("Step {:6d}: predators={} prey={} pred_births={} prey_births={} pred_deaths={} prey_deaths={} "
               "pred_species={} prey_species={}",
               state_.metrics.step, state_.metrics.predator_count, state_.metrics.prey_count,
               state_.metrics.predator_births, state_.metrics.prey_births, state_.metrics.predator_deaths,
               state_.metrics.prey_deaths, state_.metrics.predator_species, state_.metrics.prey_species);
}

bool App::run() {
  if (!cfg_.headless && !visualization_) {
    spdlog::error("Visualization requested but not initialized");
    return true;
  }

  bool completed = true;
  bool failed = false;
  while (cfg_.sim_config.max_steps == 0 || state_.runtime.step < cfg_.sim_config.max_steps) {
    MOONAI_PROFILE_SCOPE("frame_total");

    if (g_running_ == 0) {
      completed = false;
      break;
    }

    if (!cfg_.headless) {
      visualization_->handle_events();
      if (visualization_->should_close()) {
        completed = false;
        break;
      }
    }

    int steps_to_run = state_.ui.paused ? state_.ui.step_requested : std::max(1, state_.ui.speed_multiplier);
    state_.ui.step_requested = false;
    for (int i = 0; i < steps_to_run; ++i) {
      MOONAI_PROFILE_SCOPE("step");

      if (!step()) {
        failed = true;
        break;
      }

      if (state_.runtime.step % cfg_.sim_config.report_interval_steps == 0) {
        record_and_log();
      }
    }

    if (failed) {
      break;
    }

    if (!cfg_.headless) {
      visualization_->render(build_frame_snapshot(state_, cfg_));
    }
  }

  logger_.flush();

  if (failed) {
    spdlog::error("Simulation step failed");
    return false;
  }

  if (!completed) {
    spdlog::info("Simulation stopped by user (window closed)");
  }

  spdlog::info("Output saved to: {}", logger_.run_dir());

  return completed;
}

} // namespace moonai
