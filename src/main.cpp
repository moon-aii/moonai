#include "app/app.hpp"
#include "core/config.hpp"
#include "core/lua_runtime.hpp"

#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>

namespace {

struct MainArgs {
  std::string config_path = "config.lua";
  bool headless = false;
  bool verbose = false;
  bool help = false;
  std::optional<int> max_steps_override;

  std::string experiment_name;
  bool run_all = false;
  bool list_experiments = false;
  std::string run_name;
  bool validate_only = false;
};

struct ParseMainResult {
  MainArgs args;
  bool ok = true;
};

void print_main_usage(const char *program_name) {
  std::printf("MoonAI - Predator-Prey Evolutionary Simulation\n"
              "Usage: %s [OPTIONS] [config.lua]\n"
              "\n"
              "Options:\n"
              "  -c, --config <path>       Path to Lua config (default: config.lua)\n"
              "  -n, --steps <n>           Override max steps (0 = infinite)\n"
              "      --headless            Run without visualization\n"
              "  -v, --verbose             Enable debug logging\n"
              "\n"
              "Experiment orchestration:\n"
              "      --experiment <name>   Select one experiment from a multi-config Lua file\n"
              "      --all                 Run all experiments sequentially (headless only)\n"
              "      --list                List experiment names and exit\n"
              "      --name <name>         Override output directory name\n"
              "      --validate            Validate config and exit\n"
              "\n"
              "  -h, --help                Show this help message\n",
              program_name);
}

bool parse_int_arg(const char *value, int &out) {
  try {
    out = std::stoi(value);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

ParseMainResult parse_main_args(int argc, const char *argv[]) {
  ParseMainResult result;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      result.args.help = true;
      continue;
    }

    if (arg == "--headless") {
      result.args.headless = true;
    } else if (arg == "-v" || arg == "--verbose") {
      result.args.verbose = true;
    } else if (arg == "--all") {
      result.args.run_all = true;
    } else if (arg == "--list") {
      result.args.list_experiments = true;
    } else if (arg == "--validate") {
      result.args.validate_only = true;
    } else if (arg == "-n" || arg == "--steps") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
        result.ok = false;
        continue;
      }

      int parsed = 0;
      if (!parse_int_arg(argv[++i], parsed)) {
        std::fprintf(stderr, "Invalid steps value '%s'\n", argv[i]);
        result.ok = false;
        continue;
      }
      result.args.max_steps_override = parsed;
    } else if (arg == "-c" || arg == "--config") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
        result.ok = false;
        continue;
      }
      result.args.config_path = argv[++i];
    } else if (arg == "--experiment") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
        result.ok = false;
        continue;
      }
      result.args.experiment_name = argv[++i];
    } else if (arg == "--name") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
        result.ok = false;
        continue;
      }
      result.args.run_name = argv[++i];
    } else if (!arg.empty() && arg[0] != '-') {
      result.args.config_path = arg;
    } else {
      spdlog::warn("Unknown argument: {}", arg);
    }
  }

  return result;
}

bool prepare_config(moonai::SimulationConfig &config) {
  const auto validation_errors = moonai::validate_config(config);
  if (!validation_errors.empty()) {
    for (const auto &error : validation_errors) {
      spdlog::error("Config error [{}]: {}", error.field, error.message);
    }
    return false;
  }

  return true;
}

int run_experiment(const std::string &name, const moonai::SimulationConfig &config, const MainArgs &args) {
  moonai::AppConfig app_cfg;
  app_cfg.sim_config = config;
  app_cfg.experiment_name = name;
  app_cfg.headless = args.headless;
  app_cfg.run_name_override = args.run_name.empty() ? std::nullopt : std::optional(args.run_name);
  app_cfg.speed_multiplier = 1;

  if (args.max_steps_override.has_value()) {
    app_cfg.sim_config.max_steps = *args.max_steps_override;
  }

  if (!prepare_config(app_cfg.sim_config)) {
    return 1;
  }

  moonai::App app(app_cfg);
  return app.run() ? 0 : 1;
}

} // namespace

int main(int argc, const char *argv[]) {
  const auto parsed = parse_main_args(argc, argv);
  const auto &args = parsed.args;

  if (args.help) {
    print_main_usage(argv[0]);
    return parsed.ok ? 0 : 1;
  }

  if (!parsed.ok) {
    print_main_usage(argv[0]);
    return 1;
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

    if (args.max_steps_override.has_value()) {
      config.max_steps = *args.max_steps_override;
    }

    const auto validation_errors = moonai::validate_config(config);
    if (validation_errors.empty()) {
      std::printf("OK\n");
      return 0;
    }
    for (const auto &error : validation_errors) {
      std::fprintf(stderr, "ERROR [%s]: %s\n", error.field.c_str(), error.message.c_str());
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
