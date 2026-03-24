set(MOONAI_DATA_SOURCES
    logger.cpp
    metrics.cpp
)

add_library(moonai_data STATIC ${MOONAI_DATA_SOURCES})
add_library(moonai_data_profiler STATIC ${MOONAI_DATA_SOURCES})

target_include_directories(moonai_data PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(moonai_data_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_data PUBLIC
    moonai_core
)

target_link_libraries(moonai_data_profiler PUBLIC
    moonai_core_profiler
)

target_compile_definitions(moonai_data_profiler PUBLIC MOONAI_BUILD_PROFILER)
