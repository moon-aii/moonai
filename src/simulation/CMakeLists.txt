set(MOONAI_SIMULATION_SOURCES
    simulation.cpp
    common.cpp
    cpu.cpp
)

if(MOONAI_ENABLE_CUDA)
    list(APPEND MOONAI_SIMULATION_SOURCES
        gpu.cpp
        backends/cuda/gpu_data_buffer.cu
        backends/cuda/gpu_batch.cu
    )
endif()

add_library(moonai_simulation STATIC ${MOONAI_SIMULATION_SOURCES})

target_include_directories(moonai_simulation PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_simulation PUBLIC
    moonai_build_config
    moonai_core
)

if(MOONAI_ENABLE_CUDA)
    set_property(TARGET moonai_simulation PROPERTY CUDA_ARCHITECTURES native)
endif()

if(MOONAI_BUILD_PROFILER)
    add_library(moonai_simulation_profiler STATIC ${MOONAI_SIMULATION_SOURCES})
    target_include_directories(moonai_simulation_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moonai_simulation_profiler PUBLIC
        moonai_build_config
        moonai_core_profiler
    )
    target_compile_definitions(moonai_simulation_profiler PUBLIC MOONAI_BUILD_PROFILER)

    if(MOONAI_ENABLE_CUDA)
        set_property(TARGET moonai_simulation_profiler PROPERTY CUDA_ARCHITECTURES native)
    endif()
endif()
