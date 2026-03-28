set(MOONAI_SIMULATION_SOURCES
    food_store.cpp
    simulation_manager.cpp
    registry.cpp
    spatial_grid.cpp
)

add_library(moonai_simulation STATIC ${MOONAI_SIMULATION_SOURCES})
add_library(moonai_simulation_profiler STATIC ${MOONAI_SIMULATION_SOURCES})

target_include_directories(moonai_simulation PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(moonai_simulation_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_simulation PUBLIC
    moonai_core
)

target_link_libraries(moonai_simulation_profiler PUBLIC
    moonai_core_profiler
)

target_compile_definitions(moonai_simulation_profiler PUBLIC MOONAI_BUILD_PROFILER)
