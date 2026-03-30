#include "app.hpp"

#include "core/profiler_macros.hpp"
#include "data/metrics.hpp"
#include "visualization/frame_snapshot.hpp"

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

std::uint64_t App::generate_seed() {
  return static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
}

App::App(AppConfig cfg)
    : cfg_([&cfg] {
        if (cfg.sim_config.seed == 0) {
          cfg.sim_config.seed = App::generate_seed();
        }
        return std::move(cfg);
      }()),
      state_(cfg_.sim_config.seed), simulation_(cfg_.sim_config),
      evolution_(cfg_.sim_config),
      logger_(cfg_.sim_config.output_dir, cfg_.sim_config.seed,
              cfg_.run_name_override.value_or(cfg_.experiment_name)) {
  const auto errors = validate_config(cfg_.sim_config);
  if (!errors.empty()) {
    for (const auto &error : errors) {
      spdlog::error("Config error [{}]: {}", error.field, error.message);
    }
    throw std::runtime_error("Invalid simulation configuration");
  }

  state_.ui.speed_multiplier = cfg_.speed_multiplier;

  simulation_.initialize(state_);
  evolution_.initialize(state_, SensorSoA::INPUT_COUNT,
                        SensorSoA::OUTPUT_COUNT);
  evolution_.seed_initial_population(state_);
  metrics::refresh_live(state_);

  logger_.initialize(cfg_.sim_config);

  if (!cfg_.headless) {
    visualization_ =
        std::make_unique<VisualizationManager>(cfg_.sim_config, state_.ui);
    if (!visualization_->initialize()) {
      spdlog::error("Failed to initialize visualization");
      visualization_.reset();
    }
  }

  if (cfg_.enable_gpu) {
    evolution_.enable_gpu(true);
    simulation_.enable_gpu(true);
    spdlog::info("GPU acceleration enabled");
  }

  register_signal_handlers();
}

void App::step() {
  MOONAI_PROFILE_SCOPE("step");

  if (cfg_.enable_gpu && simulation_.gpu_enabled()) {
    simulation_.step_gpu(state_, evolution_);
  } else {
    simulation_.step(state_, evolution_);
  }

  for (const auto &pair : state_.runtime.pending_offspring) {
    const uint32_t child = evolution_.create_offspring(
        state_, pair.parent_a, pair.parent_b, pair.spawn_position);
    if (child != INVALID_ENTITY) {
      state_.runtime.last_step_events.push_back(
          SimEvent{SimEvent::Birth, child, child, pair.spawn_position});
      ++state_.runtime.step_events.births;
    }
  }
  state_.runtime.pending_offspring.clear();

  accumulate_step_events(state_);

  if (cfg_.sim_config.species_update_interval_steps > 0 &&
      (state_.runtime.step % cfg_.sim_config.species_update_interval_steps) ==
          0) {
    evolution_.refresh_species(state_);
  }

  ++state_.runtime.step;
  metrics::refresh_live(state_);
}

ReportMetrics App::record_and_log() {
  evolution_.refresh_species(state_);
  metrics::record_report(state_);

  logger_.log_report(state_.metrics.last_report);

  const Genome *best_genome = nullptr;
  std::size_t best_complexity = 0;
  const uint32_t entity_count = static_cast<uint32_t>(state_.registry.size());
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    const Genome *genome = moonai::genome_for(state_, idx);
    if (!genome) {
      continue;
    }

    const std::size_t complexity =
        genome->nodes().size() + genome->connections().size();
    if (!best_genome || complexity > best_complexity) {
      best_genome = genome;
      best_complexity = complexity;
    }
  }

  if (best_genome) {
    logger_.log_best_genome(state_.runtime.step, *best_genome);
  }

  logger_.log_species(state_.runtime.step, state_.evolution.species);
  logger_.flush();
  state_.runtime.report_events.clear();

  return state_.metrics.last_report;
}

bool App::should_continue() const {
  return !(g_running_ == 0) &&
         !(cfg_.sim_config.max_steps > 0 &&
           state_.runtime.step >= cfg_.sim_config.max_steps);
}

void App::log_report(const ReportMetrics &snapshot) const {
  spdlog::info(
      "Step {:6d}: predators={} prey={} births={} deaths={} species={}",
      snapshot.step, snapshot.predator_count, snapshot.prey_count,
      snapshot.births, snapshot.deaths, snapshot.num_species);
}

bool App::run() {
  if (!cfg_.headless && std::getenv("DISPLAY") == nullptr &&
      std::getenv("WAYLAND_DISPLAY") == nullptr) {
    spdlog::error("No display server found. GUI mode requires a display.");
    return false;
  }

  bool completed = true;
  bool user_quit = false;

  if (cfg_.headless) {
    while (should_continue()) {
      step();

      if (state_.runtime.step % cfg_.sim_config.report_interval_steps == 0) {
        log_report(record_and_log());
      }
    }

    if (state_.metrics.history.empty()) {
      record_and_log();
    }

    logger_.flush();
    completed = (g_running_ != 0);
  } else {
    if (!visualization_) {
      spdlog::error("Visualization requested but not initialized");
      return true;
    }

    while (should_continue()) {
      MOONAI_PROFILE_SCOPE("frame_total");

      {
        MOONAI_PROFILE_SCOPE("handle_events");
        visualization_->handle_events();
      }

      if (visualization_->should_close()) {
        completed = false;
        user_quit = true;
        break;
      }

      if (g_running_ == 0) {
        completed = false;
        break;
      }

      const bool step_requested = cfg_.interactive && state_.ui.step_requested;
      if (cfg_.interactive && state_.ui.paused && !step_requested) {
        update_selected_activations(state_);
        {
          MOONAI_PROFILE_SCOPE("render");
          visualization_->render(build_frame_snapshot(state_, cfg_));
        }
        continue;
      }

      if (cfg_.interactive && step_requested) {
        state_.ui.step_requested = false;
      }

      int steps_to_run = step_requested
                             ? 1
                             : (cfg_.interactive ? state_.ui.speed_multiplier
                                                 : cfg_.speed_multiplier);
      steps_to_run = std::max(1, steps_to_run);

      for (int i = 0; i < steps_to_run && should_continue(); ++i) {
        step();

        if (state_.runtime.step % cfg_.sim_config.report_interval_steps == 0) {
          log_report(record_and_log());
        }
      }

      if (cfg_.interactive) {
        MOONAI_PROFILE_SCOPE("update_selected");
        update_selected_activations(state_);
      }

      {
        MOONAI_PROFILE_SCOPE("render");
        visualization_->render(build_frame_snapshot(state_, cfg_));
      }
    }

    if (state_.metrics.history.empty()) {
      record_and_log();
    }

    logger_.flush();
  }

  if (!completed) {
    spdlog::info("Simulation stopped by user (window closed)");
  }

  spdlog::info("Output saved to: {}", logger_.run_dir());

  return completed;
}

} // namespace moonai
