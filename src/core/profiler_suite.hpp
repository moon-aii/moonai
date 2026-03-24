#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace moonai {

struct ProfilerSuiteConfig {
    std::string name;
    std::string config_path = "config.lua";
    std::string experiment_name;
    std::vector<std::uint64_t> seeds;
    int generations = 24;
    std::string output_dir = "output/profiles";
};

std::map<std::string, ProfilerSuiteConfig> load_profiler_suites_lua(const std::string& filepath);

} // namespace moonai
