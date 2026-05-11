cmake_minimum_required(VERSION 3.21)

# ── Project ──────────────────────────────────────────────────────────────────
project(MoonAI
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings for all build types (Debug, Release, CI, local)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wunused-variable
    -Wunused-but-set-variable
    -Wunused-function
    -Wunused-value
  )
endif()

# ── Options ──────────────────────────────────────────────────────────────────
option(MOONAI_BUILD_TESTS    "Build unit tests"                 ON)
option(MOONAI_BUILD_PROFILER "Build profiler executable"        OFF)

message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

add_library(moonai_build_config INTERFACE)

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)
find_package(sol2 CONFIG REQUIRED)
find_package(Lua REQUIRED)

target_link_libraries(moonai_build_config INTERFACE CUDA::cudart)

# ── Subsystem libraries ─────────────────────────────────────────────────────
add_subdirectory(src/core)
add_subdirectory(src/simulation)
add_subdirectory(src/evolution)
add_subdirectory(src/metrics)
add_subdirectory(src/visualization)
add_subdirectory(src/app)

# ── Main executable ──────────────────────────────────────────────────────────
add_executable(moonai src/main.cpp)

target_link_libraries(moonai PRIVATE moonai_app)

if(MOONAI_BUILD_PROFILER)
  add_executable(moonai_profiler src/profiler_main.cpp)

  target_link_libraries(moonai_profiler PRIVATE moonai_app_profiler)
  target_compile_definitions(moonai_profiler PRIVATE MOONAI_BUILD_PROFILER)
endif()

# ── Copy runtime assets to build directory ───────────────────────────────────
# config.lua is resolved relative to CWD; copy it alongside the binary so that
# running from the build dir (e.g. ./build/linux-debug/moonai) works out of the box.
set(MOONAI_BUNDLED_FONT_RELATIVE_PATH assets/fonts/JetBrainsMono-Regular.ttf)

add_custom_command(TARGET moonai POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/config.lua
        $<TARGET_FILE_DIR:moonai>/config.lua
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:moonai>/assets/fonts
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/${MOONAI_BUNDLED_FONT_RELATIVE_PATH}
        $<TARGET_FILE_DIR:moonai>/${MOONAI_BUNDLED_FONT_RELATIVE_PATH}
    COMMENT "Copying runtime assets to build directory"
)

if(MOONAI_BUILD_PROFILER)
    add_custom_command(TARGET moonai_profiler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/config.lua
            $<TARGET_FILE_DIR:moonai_profiler>/config.lua
        COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:moonai_profiler>/assets/fonts
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/${MOONAI_BUNDLED_FONT_RELATIVE_PATH}
            $<TARGET_FILE_DIR:moonai_profiler>/${MOONAI_BUNDLED_FONT_RELATIVE_PATH}
        COMMENT "Copying profiler runtime assets to build directory"
    )
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
if(MOONAI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
