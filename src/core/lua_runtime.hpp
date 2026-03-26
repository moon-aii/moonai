#pragma once

#include "core/config.hpp"

#include <map>
#include <memory>
#include <string>

namespace moonai {

struct LuaCallbacks {
  bool has_fitness_fn = false;
  bool has_on_report_window_end = false;
  bool has_on_experiment_start = false;
  bool has_on_experiment_end = false;
};

struct ReportWindowStats {
  int step;
  int window_index;
  float best_fitness;
  float avg_fitness;
  int num_species;
  int alive_predators;
  int alive_prey;
  float avg_complexity;
};

class LuaRuntime {
public:
  LuaRuntime();
  ~LuaRuntime();

  LuaRuntime(const LuaRuntime &) = delete;
  LuaRuntime &operator=(const LuaRuntime &) = delete;

  std::map<std::string, SimulationConfig>
  load_config(const std::string &filepath);

  void select_experiment(const std::string &name);

  const LuaCallbacks &callbacks() const;

  float call_fitness(float age_ratio, float kills_or_food, float energy_ratio,
                     float alive_bonus, float dist_ratio, float complexity,
                     const SimulationConfig &config);

  bool call_on_report_window_end(const ReportWindowStats &stats,
                                 std::map<std::string, float> &overrides);
  void call_on_experiment_start(const SimulationConfig &config);
  void call_on_experiment_end(const ReportWindowStats &stats);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace moonai
