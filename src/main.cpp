#include "core/config.hpp"
#include "data/logger.hpp"
#include "simulation/session.hpp"

#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>

namespace {

int run_experiment(const std::string &name, moonai::SimulationConfig config,
                   const moonai::CLIArgs &args) {
  // Check for display
  bool headless = args.headless;
  if (!headless && std::getenv("DISPLAY") == nullptr &&
      std::getenv("WAYLAND_DISPLAY") == nullptr) {
    headless = true;
    spdlog::warn("No display server found; switching to headless mode.");
  }

  // Build SessionConfig
  moonai::SessionConfig session_cfg;
  session_cfg.sim_config = config;
  session_cfg.experiment_name = name;
  session_cfg.output_dir = config.output_dir;
  session_cfg.seed = args.seed_override != 0 ? args.seed_override : config.seed;
  session_cfg.headless = headless;
  session_cfg.enable_gpu = !args.no_gpu;
  session_cfg.run_name_override =
      args.run_name.empty() ? std::nullopt : std::optional(args.run_name);
  // Default interaction settings for normal run
  session_cfg.enable_interactions = true;
  session_cfg.auto_run = false;
  session_cfg.speed_multiplier = 1;

  if (args.max_steps_override != 0) {
    session_cfg.sim_config.max_steps = args.max_steps_override;
  }

  // Create Session and run - signals handled internally
  moonai::Session session(session_cfg);
  session.run();

  spdlog::info("Output saved to: {}", session.logger().run_dir());

  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  const auto args = moonai::parse_args(argc, argv);
  if (args.help) {
    moonai::print_usage(argv[0]);
    return 0;
  }

  spdlog::set_level(args.verbose ? spdlog::level::debug : spdlog::level::info);

  auto configs = moonai::load_all_configs_lua(args.config_path);
  if (configs.empty()) {
    spdlog::error("No configs loaded from '{}'", args.config_path);
    return 1;
  }

  if (args.list_experiments) {
    for (const auto &[config_name, _] : configs) {
      std::printf("%s\n", config_name.c_str());
    }
    return 0;
  }

  if (args.validate_only) {
    moonai::SimulationConfig config;
    if (!args.experiment_name.empty()) {
      config = configs.at(args.experiment_name);
    } else {
      config = configs.begin()->second;
    }
    const auto errors = moonai::validate_config(config);
    if (errors.empty()) {
      std::printf("OK\n");
      return 0;
    }
    for (const auto &error : errors) {
      std::fprintf(stderr, "ERROR [%s]: %s\n", error.field.c_str(),
                   error.message.c_str());
    }
    return 1;
  }

  if (args.run_all) {
    int failures = 0;
    for (const auto &[config_name, config] : configs) {
      failures += run_experiment(config_name, config, args);
    }
    return failures == 0 ? 0 : 1;
  }

  std::string selected = args.experiment_name;
  if (selected.empty()) {
    selected = configs.begin()->first;
    if (configs.size() > 1) {
      spdlog::warn("Multiple experiments found; using '{}'.", selected);
    }
  }

  auto it = configs.find(selected);
  if (it == configs.end()) {
    spdlog::error("Experiment '{}' not found.", selected);
    return 1;
  }

  return run_experiment(selected, it->second, args);
}
