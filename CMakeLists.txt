cmake_minimum_required(VERSION 3.21)

# ── Project ──────────────────────────────────────────────────────────────────
project(MoonAI
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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
set(MOONAI_CUDA "AUTO" CACHE STRING "CUDA build mode: AUTO, ON, or OFF")
set_property(CACHE MOONAI_CUDA PROPERTY STRINGS AUTO ON OFF)

# ── CUDA auto-detection ───────────────────────────────────────────────────────
# AUTO uses nvcc when present. OFF forces CPU-only CI/release builds.
if(MOONAI_CUDA STREQUAL "OFF")
    set(MOONAI_ENABLE_CUDA OFF)
    message(STATUS "CUDA disabled by MOONAI_CUDA=OFF — building CPU-only.")
else()
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(MOONAI_ENABLE_CUDA ON)
        message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
    elseif(MOONAI_CUDA STREQUAL "ON")
        message(FATAL_ERROR "MOONAI_CUDA=ON but no CUDA compiler was found")
    else()
        set(MOONAI_ENABLE_CUDA OFF)
        message(STATUS "CUDA not found — building CPU-only.")
    endif()
endif()

add_library(moonai_build_config INTERFACE)

if(MOONAI_ENABLE_CUDA)
    target_compile_definitions(moonai_build_config INTERFACE MOONAI_ENABLE_CUDA)
    target_include_directories(moonai_build_config INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)
find_package(sol2 CONFIG REQUIRED)
find_package(Lua REQUIRED)

# ── Subsystem libraries ─────────────────────────────────────────────────────
add_subdirectory(src/core)
add_subdirectory(src/simulation)
add_subdirectory(src/evolution)
add_subdirectory(src/data)
add_subdirectory(src/visualization)
add_subdirectory(src/app)

# ── Main executable ──────────────────────────────────────────────────────────
add_executable(moonai src/main.cpp)

target_link_libraries(moonai PRIVATE moonai_app)

if(MOONAI_BUILD_PROFILER)
  add_executable(moonai_profiler src/profiler_main.cpp)

  target_link_libraries(moonai_profiler PRIVATE moonai_app_profiler)
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
