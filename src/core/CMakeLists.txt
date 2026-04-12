set(MOONAI_CORE_SOURCES
    app_state.cpp
    config.cpp
    lua_runtime.cpp
    random.cpp
)

add_library(moonai_core STATIC ${MOONAI_CORE_SOURCES})

target_include_directories(moonai_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_core PUBLIC
    moonai_build_config
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    sol2::sol2
    ${LUA_LIBRARIES}
)

target_include_directories(moonai_core PUBLIC ${LUA_INCLUDE_DIR})

if(MOONAI_BUILD_PROFILER)
    add_library(moonai_core_profiler STATIC ${MOONAI_CORE_SOURCES})
    target_include_directories(moonai_core_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moonai_core_profiler PUBLIC
        moonai_build_config
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        sol2::sol2
        ${LUA_LIBRARIES}
    )
    target_include_directories(moonai_core_profiler PUBLIC ${LUA_INCLUDE_DIR})
    target_compile_definitions(moonai_core_profiler PUBLIC MOONAI_BUILD_PROFILER)
endif()
