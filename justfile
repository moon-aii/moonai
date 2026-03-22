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

# Set up Python environment for analysis
[group('setup')]
setup-python:
    uv sync

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

# Run the simulation with default config
[group('run')]
run: build
    {{build-dir}}/moonai config/default_config.json

# Run with a custom config file
[group('run')]
run-config config_path: build
    {{build-dir}}/moonai {{config_path}}

# Run in headless mode (no window, max speed)
[group('run')]
run-headless: build
    {{build-dir}}/moonai --headless config/default_config.json

# Run with CPU-only inference (disable GPU even if compiled in)
[group('run')]
run-no-gpu: build
    {{build-dir}}/moonai --no-gpu config/default_config.json

# Run headless and CPU-only (useful on servers without a GPU or display)
[group('run')]
run-server: build
    {{build-dir}}/moonai --headless --no-gpu config/default_config.json

# ─── Test ───────────────────────────────────────────────────────────────────

# Run all tests
[group('test')]
test: build
    ctest --test-dir {{build-dir}} --output-on-failure

# Run tests with verbose output
[group('test')]
test-verbose: build
    ctest --test-dir {{build-dir}} --output-on-failure --verbose

# Run a specific test by name pattern
[group('test')]
test-filter pattern: build
    ctest --test-dir {{build-dir}} --output-on-failure -R {{pattern}}

# ─── Analysis ───────────────────────────────────────────────────────────────

# Plot fitness curves from a run directory
[group('analysis')]
plot run_dir="output":
    uv run python3 analysis/plot_fitness.py {{run_dir}}

# Plot fitness and save to file
[group('analysis')]
plot-save run_dir="output" output_path="output/fitness_plot.png":
    uv run python3 analysis/plot_fitness.py {{run_dir}} -o {{output_path}}

# Create experiment config files in config/experiments/
[group('analysis')]
setup-experiments:
    uv run python3 analysis/setup_experiments.py

# Validate all experiment configs
[group('analysis')]
validate-configs:
    uv run python3 analysis/validate_config.py config/experiments/*.json

# Run the full experiment matrix (all conditions × 5 seeds × 200 generations)
[group('analysis')]
run-experiments:
    #!/usr/bin/env bash
    set -e
    for cfg in config/experiments/*.json; do
        echo "==> Running condition: $cfg"
        uv run python3 analysis/run_experiments.py \
            --binary ./build/linux-release/moonai \
            --config "$cfg" \
            --seeds 42 43 44 45 46 \
            --generations 200
    done

# Generate all plots and print results summary (single post-run step)
[group('analysis')]
report:
    uv run python3 analysis/report.py

# Run full experiment pipeline: setup → validate → run → report
[group('analysis')]
experiment-pipeline:
    just setup-experiments
    just validate-configs
    just run-experiments
    just report

# Validate GPU/CPU output parity: same stats.csv results with and without GPU
[group('analysis')]
cuda-validate:
    #!/usr/bin/env bash
    set -e
    just build-type=release configure
    just build-type=release build
    echo "Running with GPU..."
    ./build/linux-release/moonai --headless --generations 5 --seed 42 > /tmp/gpu_run.txt 2>&1
    echo "Running with CPU (--no-gpu)..."
    ./build/linux-release/moonai --headless --generations 5 --seed 42 --no-gpu > /tmp/cpu_run.txt 2>&1
    echo "Results match check (stats.csv comparison):"
    diff <(ls -t output/ | head -2 | tail -1 | xargs -I{} cat "output/{}/stats.csv") \
         <(ls -t output/ | head -1 | xargs -I{} cat "output/{}/stats.csv") \
      && echo "MATCH" || echo "DIFFER (float rounding expected between GPU/CPU paths)"

# Benchmark GPU vs CPU wall-clock time on the large-population condition
[group('analysis')]
cuda-bench:
    #!/usr/bin/env bash
    just build-type=release configure
    just build-type=release build
    echo "=== GPU path (pop_large) ==="
    time ./build/linux-release/moonai \
        --headless --generations 20 --seed 42 --config config/experiments/pop_large.json
    echo "=== CPU path (pop_large) ==="
    time ./build/linux-release/moonai \
        --headless --no-gpu --generations 20 --seed 42 --config config/experiments/pop_large.json

# ─── Clean ──────────────────────────────────────────────────────────────────


# Remove build directory
[group('clean')]
clean:
    rm -rf build/

# Remove build and output directories
[group('clean')]
clean-all: clean
    rm -rf output/

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
    ./build/linux-release/moonai --headless --generations 1 --verbose \
        config/experiments/pop_large_seed1.json 2>&1 | grep -E "CPU eval|GPU eval|CUDA"

# Run visual mode briefly and capture FPS from stdout (requires display)
[group('dev')]
bench-fps: build
    timeout 10 ./build/linux-debug/moonai config/default_config.json 2>&1 | grep -i fps || true

# Profile with perf (Linux only — requires perf installed)
[group('dev')]
profile: release
    perf record -g ./build/linux-release/moonai --headless --generations 20 config/default_config.json
    perf report

# Build with AddressSanitizer and run headless for 5 generations
[group('dev')]
check-memory:
    cmake --preset linux-debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" -B build/linux-asan
    cmake --build build/linux-asan --parallel
    ./build/linux-asan/moonai --headless --generations 5 --seed 42 config/default_config.json

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
