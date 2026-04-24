add_library(moonai_app STATIC app.cpp)

target_include_directories(moonai_app PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_app PUBLIC
    moonai_build_config
    moonai_core
    moonai_metrics
    moonai_simulation
    moonai_evolution
    moonai_visualization
)

if(MOONAI_BUILD_PROFILER)
    add_library(moonai_app_profiler STATIC app.cpp)
    target_include_directories(moonai_app_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moonai_app_profiler PUBLIC
        moonai_build_config
        moonai_core_profiler
        moonai_metrics_profiler
        moonai_simulation_profiler
        moonai_evolution_profiler
        moonai_visualization_profiler
    )
    target_compile_definitions(moonai_app_profiler PUBLIC MOONAI_BUILD_PROFILER)
endif()
