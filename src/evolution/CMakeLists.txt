set(MOONAI_EVOLUTION_SOURCES
    evolution_manager.cpp
    genome.cpp
    neural_network.cpp
    species.cpp
    mutation.cpp
    crossover.cpp
    network_cache.cpp
)

if(MOONAI_ENABLE_CUDA)
    list(APPEND MOONAI_EVOLUTION_SOURCES backends/cuda/gpu_network_cache.cu)
endif()

add_library(moonai_evolution STATIC ${MOONAI_EVOLUTION_SOURCES})

target_include_directories(moonai_evolution PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_evolution PUBLIC
    moonai_build_config
    moonai_core
    moonai_simulation
)

if(MOONAI_ENABLE_CUDA)
    set_property(TARGET moonai_evolution PROPERTY CUDA_ARCHITECTURES native)
endif()

if(MOONAI_BUILD_PROFILER)
    add_library(moonai_evolution_profiler STATIC ${MOONAI_EVOLUTION_SOURCES})
    target_include_directories(moonai_evolution_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moonai_evolution_profiler PUBLIC
        moonai_build_config
        moonai_core_profiler
        moonai_simulation_profiler
    )
    target_compile_definitions(moonai_evolution_profiler PUBLIC MOONAI_BUILD_PROFILER)

    if(MOONAI_ENABLE_CUDA)
        set_property(TARGET moonai_evolution_profiler PROPERTY CUDA_ARCHITECTURES native)
    endif()
endif()
