#include "core/profiler_suite.hpp"

#include <spdlog/spdlog.h>

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace moonai {

std::map<std::string, ProfilerSuiteConfig> load_profiler_suites_lua(const std::string& filepath) {
    std::map<std::string, ProfilerSuiteConfig> suites;

    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::table, sol::lib::string);

    try {
        sol::protected_function_result result = lua.safe_script_file(filepath);
        if (!result.valid()) {
            sol::error err = result;
            spdlog::error("Lua profiler config error in '{}': {}", filepath, err.what());
            return suites;
        }

        sol::object obj = result;
        if (obj.get_type() != sol::type::table) {
            spdlog::error("Lua profiler config '{}' must return a table", filepath);
            return suites;
        }

        sol::table root = obj.as<sol::table>();
        for (auto& [key, value] : root) {
            if (key.get_type() != sol::type::string || value.get_type() != sol::type::table) {
                continue;
            }

            ProfilerSuiteConfig suite;
            suite.name = key.as<std::string>();
            const sol::table tbl = value.as<sol::table>();

            if (auto entry = tbl["config_path"]; entry.valid()) {
                suite.config_path = entry.get<std::string>();
            }
            if (auto entry = tbl["experiment"]; entry.valid()) {
                suite.experiment_name = entry.get<std::string>();
            }
            if (auto entry = tbl["generations"]; entry.valid()) {
                suite.generations = entry.get<int>();
            }
            if (auto entry = tbl["output_dir"]; entry.valid()) {
                suite.output_dir = entry.get<std::string>();
            }
            const sol::object seeds_obj = tbl["seeds"];
            if (seeds_obj.valid() && seeds_obj.get_type() == sol::type::table) {
                const sol::table seeds_tbl = seeds_obj.as<sol::table>();
                for (auto& [_, seed_value] : seeds_tbl) {
                    if (!seed_value.valid()) {
                        continue;
                    }
                    suite.seeds.push_back(static_cast<std::uint64_t>(seed_value.as<double>()));
                }
            }

            if (suite.experiment_name.empty()) {
                spdlog::warn("Profiler suite '{}' is missing required field 'experiment'", suite.name);
                continue;
            }
            if (suite.seeds.empty()) {
                spdlog::warn("Profiler suite '{}' has no seeds", suite.name);
                continue;
            }

            suites[suite.name] = std::move(suite);
        }

        if (suites.empty()) {
            spdlog::error("Lua profiler config '{}' returned no named suites.", filepath);
        } else {
            spdlog::info("Loaded {} profiler suite(s) from '{}'.", suites.size(), filepath);
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to load Lua profiler config '{}': {}", filepath, e.what());
    }

    return suites;
}

} // namespace moonai
