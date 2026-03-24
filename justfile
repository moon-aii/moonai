# MoonAI - Project Commands
# Usage: just <recipe>
# Run `just --list` to see all available recipes.

# Default recipe: build debug
default: build

# ─── Configuration ──────────────────────────────────────────────────────────

# Detect OS
os := if os() == "windows" { "windows" } else { "linux" }

# vcpkg root (falls back to ~/.vcpkg if not set in environment)
export VCPKG_ROOT := env("VCPKG_ROOT", env("HOME") + "/.vcpkg")

# Build type (override with: just build-type=release build)
build-type := "debug"

preset := os + "-" + build-type
build-dir := "build" / preset

# ─── Setup ──────────────────────────────────────────────────────────────────

# Install vcpkg and bootstrap it
[group('setup')]
setup-vcpkg:
    #!/usr/bin/env bash
    if [ -z "$VCPKG_ROOT" ]; then
        echo "VCPKG_ROOT is not set. Installing vcpkg to ~/.vcpkg..."
        git clone https://github.com/microsoft/vcpkg.git ~/.vcpkg
        ~/.vcpkg/bootstrap-vcpkg.sh
        echo "Add to your shell profile:"
        echo '  export VCPKG_ROOT="$HOME/.vcpkg"'
        echo '  export PATH="$VCPKG_ROOT:$PATH"'
    else
        echo "vcpkg found at $VCPKG_ROOT"
        echo "Bootstrapping..."
        "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
    fi

# Install system dependencies (Arch Linux)
[group('setup')]
setup-arch:
    sudo pacman -S --needed cmake ninja gcc sfml cuda python-pandas python-matplotlib

# Install system dependencies (Ubuntu/Debian)
[group('setup')]
setup-ubuntu:
    sudo apt-get update && sudo apt-get install -y \
        cmake ninja-build g++ \
        libsfml-dev \
        nvidia-cuda-toolkit \
        python3-pandas python3-matplotlib

# Set up Python environments for simulation and profiler analysis
[group('setup')]
setup-python:
    cd analysis && uv sync
    cd profiler && uv sync

# Full first-time setup
[group('setup')]
setup: setup-vcpkg
    @echo "Setup complete. Run 'just configure' next."

# ─── Build ──────────────────────────────────────────────────────────────────

# Configure CMake (run after setup or when CMakeLists change)
[group('build')]
configure:
    cmake --preset {{preset}}

# Build the project
[group('build')]
build:
    cmake --build {{build-dir}} --parallel

# Build in release mode
[group('build')]
release:
    just build-type=release configure
    just build-type=release build

# ─── Run ────────────────────────────────────────────────────────────────────

# Run the simulation with default config (auto-selects the 'default' experiment)
[group('run')]
run: build
    {{build-dir}}/moonai config.lua --experiment default

# Run with a custom config file
[group('run')]
run-config config_path: build
    {{build-dir}}/moonai {{config_path}}

# Run in headless mode (no window, max speed)
[group('run')]
run-headless: build
    {{build-dir}}/moonai config.lua --experiment default --headless

# Run with CPU-only inference (disable GPU even if compiled in)
[group('run')]
run-no-gpu: build
    {{build-dir}}/moonai config.lua --experiment default --no-gpu

# Run headless and CPU-only (useful on servers without a GPU or display)
[group('run')]
run-server: build
    {{build-dir}}/moonai config.lua --experiment default --headless --no-gpu

# Validate a config file
[group('run')]
validate config_path="config.lua": build
    {{build-dir}}/moonai --validate {{config_path}}

# ─── Test ───────────────────────────────────────────────────────────────────

# Run all tests
[group('test')]
test: build
    ctest --test-dir {{build-dir}} --output-on-failure

# Run tests with verbose output
[group('test')]
test-verbose: build
    ctest --test-dir {{build-dir}} --output-on-failure --verbose

# Run only CUDA/GPU tests
[group('test')]
test-gpu: build
    ctest --test-dir {{build-dir}} --output-on-failure -R GpuTest

# Run a specific test by name pattern
[group('test')]
test-filter pattern: build
    ctest --test-dir {{build-dir}} --output-on-failure -R {{pattern}}

# ─── Experiments ─────────────────────────────────────────────────────────────

# List all experiments defined in the Lua config
[group('experiment')]
list-experiments: build
    {{build-dir}}/moonai config.lua --list

# Run the full experiment matrix (all conditions × seeds, headless)
[group('experiment')]
experiments: release
    ./build/linux-release/moonai config.lua --all --headless

# Run one named experiment (headless)
[group('experiment')]
run-experiment name: release
    ./build/linux-release/moonai config.lua --experiment {{name}} --headless

# ─── Analysis ───────────────────────────────────────────────────────────────

# Generate the self-contained HTML analysis report from output/
[group('analysis')]
analyse:
    cd analysis && uv run moonai-analysis

# Generate the self-contained HTML profiler report from output/profiles/
[group('analysis')]
analyse-profile:
    cd profiler && uv run moonai-profiler

# Full experiment pipeline: run all experiments → generate report
[group('experiment')]
experiment-pipeline: experiments analyse

# Validate GPU/CPU output parity: same stats.csv results with and without GPU
[group('gpu')]
gpu-validate:
    #!/usr/bin/env bash
    set -e
    just build-type=release configure
    just build-type=release build
    echo "Running with GPU..."
    ./build/linux-release/moonai config.lua --experiment default --headless -g 5 --seed 42 > /tmp/gpu_run.txt 2>&1
    echo "Running with CPU (--no-gpu)..."
    ./build/linux-release/moonai config.lua --experiment default --headless --no-gpu -g 5 --seed 42 > /tmp/cpu_run.txt 2>&1
    echo "Results match check (stats.csv comparison):"
    diff <(ls -t output/ | head -2 | tail -1 | xargs -I{} cat "output/{}/stats.csv") \
         <(ls -t output/ | head -1 | xargs -I{} cat "output/{}/stats.csv") \
      && echo "MATCH" || echo "DIFFER (float rounding expected between GPU/CPU paths)"

# Benchmark GPU vs CPU wall-clock time on the large-population condition
[group('gpu')]
gpu-bench:
    #!/usr/bin/env bash
    just build-type=release configure
    just build-type=release build
    echo "=== GPU path (pop_large) ==="
    time ./build/linux-release/moonai config.lua \
        --experiment pop_large_seed42 --headless -g 20
    echo "=== CPU path (pop_large) ==="
    time ./build/linux-release/moonai config.lua \
        --experiment pop_large_seed42 --headless --no-gpu -g 20

# ─── Clean ──────────────────────────────────────────────────────────────────


# Remove build directory
[group('clean')]
clean:
    rm -rf build/

# Remove build, output, and all generated report artifacts
[group('clean')]
clean-all: clean
    rm -rf output/
    rm -rf analysis/output/
    rm -rf profiler/output/

# ─── Development ────────────────────────────────────────────────────────────

# Format all C++ source files (requires clang-format)
[group('dev')]
format:
    find src tests -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' | \
        xargs clang-format -i --style=file

# Generate compile_commands.json for IDE/LSP integration
[group('dev')]
compdb:
    cmake --preset {{preset}} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    ln -sf {{build-dir}}/compile_commands.json compile_commands.json

# Check for common issues (requires cppcheck)
[group('dev')]
lint:
    cppcheck --enable=warning,style,performance --std=c++17 \
        --suppress=missingInclude --quiet src/

# Benchmark NN forward pass: 1 generation of pop_large, verbose timing
[group('dev')]
bench-nn: release
    ./build/linux-release/moonai config.lua \
        --experiment pop_large_seed42 --headless -g 1 -v 2>&1 | grep -E "CPU eval|GPU eval|CUDA"

# Run the built-in profiler on the baseline experiment
[group('dev')]
profile: release
    ./build/linux-release/moonai_profiler profiler.lua \
        --suite baseline

# Full profiler pipeline: run profiler -> generate profiler report
[group('dev')]
profile-pipeline: profile analyse-profile

# Run visual mode briefly and capture FPS from stdout (requires display)
[group('dev')]
bench-fps: build
    timeout 10 ./build/linux-debug/moonai config.lua --experiment default 2>&1 | grep -i fps || true

# Build with AddressSanitizer and run headless for 5 generations
[group('dev')]
check-memory:
    cmake --preset linux-debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" -B build/linux-asan
    cmake --build build/linux-asan --parallel
    ./build/linux-asan/moonai config.lua --experiment default --headless -g 5 --seed 42

# ─── Info ───────────────────────────────────────────────────────────────────

# Show project info and detected configuration
[group('info')]
info:
    @echo "MoonAI Project"
    @echo "─────────────────────────────"
    @echo "OS:         {{os}}"
    @echo "Build type: {{build-type}}"
    @echo "Preset:     {{preset}}"
    @echo "Build dir:  {{build-dir}}"
    @echo "VCPKG_ROOT: ${VCPKG_ROOT:-NOT SET}"
