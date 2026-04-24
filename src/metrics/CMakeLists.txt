set(MOONAI_METRICS_SOURCES
    logger.cpp
    metrics.cpp
)

add_library(moonai_metrics STATIC ${MOONAI_METRICS_SOURCES})

target_include_directories(moonai_metrics PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_metrics PUBLIC
    moonai_build_config
    moonai_core
    moonai_evolution
)

if(MOONAI_BUILD_PROFILER)
    add_library(moonai_metrics_profiler STATIC ${MOONAI_METRICS_SOURCES})
    target_include_directories(moonai_metrics_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moonai_metrics_profiler PUBLIC
        moonai_build_config
        moonai_core_profiler
        moonai_evolution_profiler
    )
    target_compile_definitions(moonai_metrics_profiler PUBLIC MOONAI_BUILD_PROFILER)
endif()
