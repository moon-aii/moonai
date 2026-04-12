set(MOONAI_DATA_SOURCES
    logger.cpp
    metrics.cpp
)

add_library(moonai_data STATIC ${MOONAI_DATA_SOURCES})

target_include_directories(moonai_data PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_data PUBLIC
    moonai_build_config
    moonai_core
    moonai_evolution
)

if(MOONAI_BUILD_PROFILER)
    add_library(moonai_data_profiler STATIC ${MOONAI_DATA_SOURCES})
    target_include_directories(moonai_data_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moonai_data_profiler PUBLIC
        moonai_build_config
        moonai_core_profiler
        moonai_evolution_profiler
    )
    target_compile_definitions(moonai_data_profiler PUBLIC MOONAI_BUILD_PROFILER)
endif()
