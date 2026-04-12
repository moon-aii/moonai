#pragma once

#include "core/config.hpp"

#include <map>
#include <memory>
#include <string>

namespace moonai {

std::map<std::string, SimulationConfig> load_all_configs_lua(const std::string &filepath);

class LuaRuntime {
public:
  LuaRuntime();
  ~LuaRuntime();

  LuaRuntime(const LuaRuntime &) = delete;
  LuaRuntime &operator=(const LuaRuntime &) = delete;

  std::map<std::string, SimulationConfig> load_config(const std::string &filepath);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace moonai
