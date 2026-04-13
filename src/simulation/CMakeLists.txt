set(MOONAI_SIMULATION_SOURCES
    simulation.cpp
    buffers.cu
    batch.cu
)

add_library(moonai_simulation STATIC ${MOONAI_SIMULATION_SOURCES})

target_include_directories(moonai_simulation PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_simulation PUBLIC
    moonai_build_config
    moonai_core
)

set_property(TARGET moonai_simulation PROPERTY CUDA_ARCHITECTURES native)

if(MOONAI_BUILD_PROFILER)
    add_library(moonai_simulation_profiler STATIC ${MOONAI_SIMULATION_SOURCES})
    target_include_directories(moonai_simulation_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moonai_simulation_profiler PUBLIC
        moonai_build_config
        moonai_core_profiler
    )
    target_compile_definitions(moonai_simulation_profiler PUBLIC MOONAI_BUILD_PROFILER)

    set_property(TARGET moonai_simulation_profiler PROPERTY CUDA_ARCHITECTURES native)
endif()
