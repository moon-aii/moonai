set(MOONAI_EVOLUTION_SOURCES
    evolution_manager.cpp
    genome.cpp
    neural_network.cpp
    species.cpp
    mutation.cpp
    crossover.cpp
)

add_library(moonai_evolution STATIC ${MOONAI_EVOLUTION_SOURCES})
add_library(moonai_evolution_profiler STATIC ${MOONAI_EVOLUTION_SOURCES})

target_include_directories(moonai_evolution PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(moonai_evolution_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_evolution PUBLIC
    moonai_core
    moonai_simulation
)

target_link_libraries(moonai_evolution_profiler PUBLIC
    moonai_core_profiler
    moonai_simulation_profiler
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(moonai_evolution PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(moonai_evolution PUBLIC MOONAI_OPENMP_ENABLED)
    target_link_libraries(moonai_evolution_profiler PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(moonai_evolution_profiler PUBLIC MOONAI_OPENMP_ENABLED)
endif()

target_compile_definitions(moonai_evolution_profiler PUBLIC MOONAI_BUILD_PROFILER)
