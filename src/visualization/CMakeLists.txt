add_library(moonai_visualization STATIC
    visualization_manager.cpp
    renderer.cpp
    overlay.cpp
    frame_snapshot.cpp
)

target_include_directories(moonai_visualization PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_visualization PUBLIC
    moonai_build_config
    moonai_core
    moonai_evolution
    moonai_simulation
    SFML::Graphics
    SFML::Window
    SFML::System
)

if(MOONAI_BUILD_PROFILER)
    add_library(moonai_visualization_profiler STATIC
        visualization_manager.cpp
        renderer.cpp
        overlay.cpp
        frame_snapshot.cpp
    )
    target_include_directories(moonai_visualization_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moonai_visualization_profiler PUBLIC
        moonai_build_config
        moonai_core_profiler
        moonai_evolution_profiler
        moonai_simulation_profiler
        SFML::Graphics
        SFML::Window
        SFML::System
    )

    target_compile_definitions(moonai_visualization_profiler PUBLIC MOONAI_BUILD_PROFILER)
endif()
