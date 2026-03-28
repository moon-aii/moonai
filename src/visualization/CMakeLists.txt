add_library(moonai_visualization STATIC
    visualization_manager.cpp
    renderer.cpp
    ui_overlay.cpp
)

add_library(moonai_visualization_profiler STATIC
    visualization_manager.cpp
    renderer.cpp
    ui_overlay.cpp
)

target_include_directories(moonai_visualization PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(moonai_visualization_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_visualization PUBLIC
    moonai_core
    moonai_simulation
    SFML::Graphics
    SFML::Window
    SFML::System
)

target_link_libraries(moonai_visualization_profiler PUBLIC
    moonai_core_profiler
    moonai_simulation_profiler
    SFML::Graphics
    SFML::Window
    SFML::System
)

target_compile_definitions(moonai_visualization_profiler PUBLIC MOONAI_BUILD_PROFILER)
