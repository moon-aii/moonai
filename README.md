# MoonAI

A modular and extensible simulation platform for studying continuous evolutionary algorithms and neural network evolution through predator-prey dynamics.

**CMPE 491/492 - Senior Design Project | TED University**

**Website:** https://moon-aii.github.io/moonai/

## Overview

MoonAI uses a simplified predator-prey environment as a synthetic benchmark to evaluate evolutionary computation methods. Agents (predators and prey) are controlled by neural networks whose structure and weights evolve continuously through births and deaths using the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm.

The platform enables researchers to:

- Observe how neural network topologies emerge and grow in complexity through evolution
- Compare different genetic representations, mutation strategies, and selection methods
- Generate structured datasets for machine learning research without real-world data
- Visualize agent behavior and algorithm evolution in real time

## Performance

MoonAI achieves high performance through data-oriented ECS architecture:

**Key Optimizations:**
- **Cache-friendly layouts**: Structure-of-Arrays (SoA) component storage
- **Efficient GPU packing**: Contiguous memcpy from ECS to GPU buffers
- **Parallel systems**: OpenMP parallelization across all simulation systems
- **SIMD-ready**: Contiguous data enables AVX/AVX-512 vectorization

## Key Features

- **Entity-Component-System Architecture** - Data-oriented design with sparse-set ECS, cache-friendly SoA memory layouts, and 5-10x performance improvement
- **Clean GPU Abstraction** - ECS data efficiently packed into GPU buffers; kernels consume contiguous buffers (decoupled architecture)
- **NEAT Implementation** - Evolves both topology and weights of neural networks simultaneously
- **Real-Time Visualization** - SFML-based rendering with interactive controls and live NN activation display
- **GPU Acceleration** - CUDA backend for sensing, neural inference, and simulation systems on GPU at large populations; available in both visual and headless modes with runtime CPU fallback
- **Cross-Platform** - Runs on Linux and Windows with matched features and stable runtime behavior
- **Reproducible Experiments** - Seeded RNG with deterministic behavior within each execution backend; CPU and GPU runs are kept numerically close but are not bit-exact twins
- **Lua Configuration** - Define named experiments and parameter sweeps in `config.lua` without recompilation
- **Data Export** - CSV/JSON output (including optional per-step trajectories) compatible with Python analysis tools

## Architecture

MoonAI uses a **hybrid ECS/OOP architecture** optimized for evolutionary simulation:

### Core Philosophy

- **ECS for Simulation**: Agent state, physics, and interactions use data-oriented ECS for cache efficiency and GPU compatibility
- **OOP for Evolution**: NEAT algorithms (Genome, NeuralNetwork) remain object-oriented due to complex graph mutations and variable topology
- **Clean Boundaries**: Well-defined interfaces between ECS simulation core and OOP evolution systems

### Why ECS?

Traditional OOP with `vector<unique_ptr<Agent>>` causes:
- Cache misses from pointer chasing
- Virtual dispatch overhead  
- Expensive GPU upload (field-by-field extraction)

ECS solves these with:
- Contiguous component arrays (Structure of Arrays)
- Direct GPU memory mapping (zero-copy transfers)
- Trivial parallelization (OpenMP)

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization (SFML)                     │
│              Renders agents, grid, UI overlays              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Queries ECS Components
┌──────────────────────────┴──────────────────────────────────┐
│              ECS Simulation Core (Data-Oriented)            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  Registry   │ │  Systems    │ │  GpuDataBuffer      │   │
│  │ (Components)│ │ (Logic)     │ │ (Buffer Abstraction)│   │
│  └──────┬──────┘ └──────┬──────┘ └──────────┬──────────┘   │
│         └───────────────┴───────────────────┘              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Genome References
┌──────────────────────────┴──────────────────────────────────┐
│                    Evolution Core (NEAT)                    │
│     Genome, NN, Species, Mutation, Crossover (OOP)          │
└──────────────────────────┬──────────────────────────────────┘
                           │ Exports Metrics
┌──────────────────────────┴──────────────────────────────────┐
│                    Data Management                          │
│              Logger (CSV), Metrics, Config (JSON)           │
└─────────────────────────────────────────────────────────────┘
```

| Subsystem | Pattern | Library | Description |
|-----------|---------|---------|-------------|
| `src/core/` | OOP | `moonai_core` | Shared types (`Vec2`, `Entity`), Lua config loader (sol2), seeded RNG |
| `src/simulation/` | **ECS** | `moonai_simulation` | Sparse-set registry, SoA components, systems (movement, sensors, combat, energy), spatial grid |
| `src/evolution/` | OOP | `moonai_evolution` | NEAT genome, neural network, NetworkCache, speciation, mutation, crossover |
| `src/visualization/` | OOP | `moonai_visualization` | SFML window, renderer, UI overlay (queries ECS registry directly) |
| `src/data/` | OOP | `moonai_data` | CSV logger, metrics collector |
| `src/gpu/` | Mixed | `moonai_gpu` | CUDA kernels, GpuDataBuffer abstraction, ECS-to-GPU packing |

## Prerequisites

| Tool | Version | Required |
|------|---------|----------|
| C++ Compiler | C++17 support (GCC 9+, Clang 10+, MSVC 2019+) | Yes |
| CMake | 3.21+ | Yes |
| Ninja | any | Recommended |
| vcpkg | latest | Yes |
| just | any | Recommended |
| SFML | 3.x | Yes (via vcpkg) |
| CUDA Toolkit | 11.0+ | Optional (auto-detected) |
| Python | 3.10+ with uv | For analysis only |

## Quick Start

### 1. Clone and enter the project

```bash
git clone https://github.com/moon-aii/moonai.git
cd moonai
```

### 2. Install vcpkg (if not already installed)

```bash
git clone https://github.com/microsoft/vcpkg.git ~/.vcpkg
~/.vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT="$HOME/.vcpkg"  # Add to your shell profile
```

### 3. Configure and build

```bash
just configure
just build
```

Or manually:
```bash
cmake --preset linux-debug
cmake --build build/linux-debug --parallel
```

### 4. Run tests

```bash
just test              # basic run
just test --verbose    # verbose output
just test -R GpuTest   # filter tests
```

### 5. Run the simulation

```bash
just run
```

## Build

There is one build type — it always bundles SFML visualization and auto-detects CUDA:

| Command | Description |
|---------|-------------|
| `just build` | Debug build |
| `just release` | Optimized release build |

CUDA is compiled in automatically when `nvcc` is found. On machines without the CUDA Toolkit, the build succeeds and uses the CPU path.
Official GitHub releases are CPU-only; CUDA support is available from source builds on CUDA-capable machines.

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `MOONAI_BUILD_TESTS` | `ON` | Build unit tests |

### Runtime Modes

Mode selection happens at runtime via flags — no need to rebuild:

| Command | Description |
|---------|-------------|
| `just run` | Default: visualization window, GPU if available for large populations |
| `just run-release` | Release build: faster execution for large experiments |

Both commands accept additional arguments after `--`:

| Example | Description |
|---------|-------------|
| `just run -- --headless` | No window, max speed (auto-switches if `$DISPLAY` unset) |
| `just run -- --no-gpu` | Force CPU inference even if CUDA is compiled in |
| `just run -- --headless --no-gpu` | Headless + CPU-only (for servers without a display or GPU) |
| `just run -- --experiment <name>` | Run specific experiment instead of default |

CUDA is enabled at runtime when available. The GPU path executes sensing, neural inference, and simulation systems (movement, combat, energy, aging) entirely on GPU. If any GPU operation fails, MoonAI automatically disables the CUDA path and falls back to CPU execution.

### Visualization Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `↑` / `↓` or `+` / `-` | Increase / decrease simulation speed |
| `.` | Step one step (while paused) |
| `E` | Open experiment selector (multi-config only) |
| `R` | Reset simulation |
| `S` | Save screenshot |
| `Esc` | Quit |
| Left-click | Select an agent (shows stats + live NN panel) |
| Right-click drag | Pan camera |
| Scroll wheel | Zoom |

When an agent is selected, its **vision range** (semi-transparent circle), **sensor lines** (connections to nearby agents and food), and **stats panel** (bottom-left) are automatically displayed. The agent controller currently receives 12 inputs: nearest predator/prey/food as wrapped normalized `dx, dy`, energy, velocity `x/y`, and local predator/prey/food density. The **Network panel** (top-right) shows its topology with nodes colored by live activation value: blue (inactive, −1) → gray (zero) → orange (active, +1).

## Configuration

Configuration uses a single **`config.lua`** file at the project root. It returns a named table of experiments — every entry is a fully-specified run. The runtime injects C++ struct defaults as the `moonai_defaults` global (2000 agents on a 3000×3000 square world), so Lua only needs to override what it changes.

### `config.lua` structure

```lua
-- moonai_defaults is injected by the runtime (mirrors C++ SimulationConfig defaults)
-- Defaults: 500 predators, 1500 prey (2000 total), 3000×3000 square world, 1500 steps per report window
local function extend(t, overrides) ... end

-- Helper: scale world and food proportionally to population
local function scale_base(pred, prey)
    local total = pred + prey
    local default_total = moonai_defaults.predator_count + moonai_defaults.prey_count
    local factor = math.sqrt(total / default_total)
    return {
        predator_count = pred, prey_count = prey,
        grid_size = math.floor(moonai_defaults.grid_size * factor),
        food_count = math.floor(moonai_defaults.food_count * (total / default_total)),
    }
end

local conditions = {
    baseline = moonai_defaults,
    scale_5k = extend(moonai_defaults, scale_base(1250, 3750)),
    -- ...
}
local seeds = { 42, 43, 44, 45, 46 }

local experiments = {}
for name, cfg in pairs(conditions) do
    for _, seed in ipairs(seeds) do
        experiments[name .. "_seed" .. seed] = extend(cfg, { seed = seed, max_generations = 200 })
    end
end

experiments["default"] = moonai_defaults  -- auto-selected by 'just run'
return experiments
```

A single-entry file auto-selects without `--experiment`. The `default` entry (2000 agents) serves as the everyday run config with GPU auto-enabled.

### CLI flags

| Flag | Purpose |
|------|---------|
| `--experiment <name>` | Select one experiment by name |
| `--all` | Run all experiments sequentially (headless only) |
| `--list` | List experiment names and exit |
| `--name <name>` | Override output directory name |
| `--validate` | Load + validate config, print result, exit |
| `--set key=value` | Override any param after Lua load (repeatable) |

### Examples

```bash
just run                                              # GUI with default config
just run -- --headless                                # Headless mode
just run -- --no-gpu                                  # Force CPU-only
just run -- --headless --no-gpu                       # Server mode (no display/GPU)
just run -- --experiment mut_low_seed42 --headless    # One experiment
just run-release -- --all --headless                  # Full batch (release build)
just run -- --set mutation_rate=0.1                   # Ad-hoc override
```

Set `seed` to `0` for random seed, or a fixed value for reproducible experiments.

## Running Experiments

### Quick start (full pipeline)

```bash
just experiment             # runs all experiments + generates report
```

### Step by step

**1. Build release binary**
```bash
just release
```

**2. List available experiments**
```bash
just list-experiments       # shows all experiments in config.lua
```

**3. Run experiments**
```bash
just experiment-run         # 66 conditions × 5 seeds × 200 report windows → output/
```

**4. Set up Python and generate analysis**
```bash
just setup-python           # installs simulation + profiler analysis dependencies via uv
just experiment-analyse     # reads output/, writes a self-contained HTML report
```

### Analysis

The Python analysis tool has a single mode: it always generates one self-contained HTML report for all qualifying runs in `output/`.

```bash
just experiment-analyse
```

Internally this runs the packaged analysis entry point from `analysis/`:

```bash
cd analysis && uv run moonai-analysis
```

The analysis step is non-interactive and always writes a timestamped report to `analysis/output/`, for example `report_20260324_154233.html`.

The generated HTML is fully self-contained: it embeds all plots and report data directly into a single file, including:

- per-condition plots for population, species, complexity, and representative-genome topology
- cross-condition comparison plots using seed-aggregated statistics
- the grouped summary table at the final sampled generation
- skipped-run information for incomplete or invalid runs
- inline styling and navigation so the report opens directly in a browser without side files

The analysis code is structured as a small package under `analysis/moonai_analysis/`:

- `pipeline.py` orchestrates the full analysis run
- `io.py` discovers runs and loads CSV/JSON data
- `labels.py` groups runs into experiment conditions
- `plots.py` generates embedded per-condition and comparison figures
- `genome.py` renders embedded representative-genome topology diagrams
- `summary.py` prepares structured summary data for the report
- `html_report.py` renders the final self-contained HTML document
- `templates/report.html.j2` defines the HTML report layout

### Profiler output and analysis

Profiler runs use the dedicated `moonai_profiler` entry point. The profiler is configured via CLI arguments instead of a config file.

```bash
just profile-run                                     # Run with defaults (600 frames, 6 seeds)
just profile-run --frames 300                        # Custom frame count
just profile-run --name mytest --output-dir results  # Custom name and output
just profile-run --frames 300 --no-gpu               # Custom frame count, disable GPU
```

**CLI Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--frames N` | 600 | Number of frames to capture per run |
| `--name <name>` | profile | Experiment name (used in output filename) |
| `--output-dir <path>` | output/profiles | Output directory |
| `--no-gpu` | false | Disable GPU acceleration |

Each profiler run writes a single JSON file to `output/profiles/`:

| File | Contents |
|------|----------|
| `YYYY-MM-DD_HH-MM-SS_name.json` | Suite manifest with per-frame timing data from all seeds |

The profiler drops the fastest and slowest runs by average frame time, and reports aggregate timing data from the remaining runs. Standard simulation builds do not include profiler instrumentation.

To generate the standalone profiler report:

```bash
just profile-analyse
```

The profiler writes a timestamped self-contained HTML report to `profiler/output/`, for example `profile_report_20260324_154233.html`.

The profiler package lives under `profiler/moonai_profiler/` and includes:

- `pipeline.py` for orchestration
- `io.py` for discovering and validating `profile_suite.json` runs
- `plots.py` for embedded timing charts
- `html_report.py` for rendering
- `templates/report.html.j2` for layout

### Experiment conditions

66 conditions defined in `config.lua` across 9 groups, each × 5 seeds = **330 deterministic runs**.

The default baseline is 2000 agents (500 predators, 1500 prey) on a 3000×3000 square world with 1500 steps per report window. Scaled experiments use `scale_base()` to maintain agent density by proportionally adjusting world size and food count.

- Group A — Baseline sweeps (2K agents)
- Group B — Scale experiments (proportional world)
- Group C — Parameter sweeps at 5K
- Group D — Parameter sweeps at 10K
- Group E — World density (5K agents, varying world size)
- Group F — Reporting window length
- Group G — Energy / resource dynamics
- Group H — Agent speed / interaction range (5K)
- Group I — Topology complexity

### Large-scale experiments

Experiments with 5K+ agents require significant compute. Recommendations:

- **GPU strongly recommended** for populations >= 2000 (auto-enabled when CUDA is available)
- **Release build** (`just release`) for 2-5x faster simulation
- **Headless mode** (`--headless`) disables rendering for maximum throughput
- **Visual mode** uses the same GPU acceleration as headless when CUDA is available, but headless mode achieves higher throughput because it eliminates rendering overhead (SFML runs on CPU, GPU is fully dedicated to simulation)
- **Memory**: ~4 GB RAM for 10K agents, ~8 GB for 20K agents
- **VRAM**: ~512 MB for 10K agents, ~1 GB for 20K agents
- Running all 330 experiments sequentially takes significant time; use `--experiment` to run specific conditions or parallelize across machines

## Development

```bash
# Generate compile_commands.json for your IDE/LSP
just compdb

# Run tests
just test              # basic run
just test --verbose    # verbose output
just test -R GpuTest   # filter tests

# Run the profiler with default settings (600 frames, 6 seeds)
just profile-run

# Run profiler with custom frame count
just profile-run --frames 300

# Generate the standalone profiler HTML report
just profile-analyse

# Full profiler pipeline: run profiler and build the report
just profile

```

The dedicated profiler writes one `profile_suite.json` file per suite under a unique
directory in `output/profiles/` by default when invoked through `just profile-run`.

## Project Structure

```
moonai/
├── CMakeLists.txt              # Root CMake configuration
├── CMakePresets.json            # Build presets for Linux/Windows
├── vcpkg.json                  # Dependency manifest
├── justfile                    # Project commands (run `just --list` for full list)
├── config.lua                  # Unified config: default run + experiment matrix (66 × 5 seeds)
├── src/
│   ├── main.cpp                # Entry point: CLI parsing, init, main loop, shutdown
│   ├── core/                   # Shared types (Vec2, Entity), config loader, Lua runtime, seeded RNG
│   ├── simulation/             # ECS CORE - Data-oriented simulation
│   │   ├── registry.hpp/cpp    # Sparse-set ECS registry with SoA storage
│   │   ├── entity.hpp          # Entity handles (index + generation)
│   │   ├── components.hpp      # SoA component definitions
│   │   ├── spatial_grid.hpp/cpp  # Entity-based spatial indexing
│   │   ├── simulation_manager.hpp/cpp # Coordinates ECS systems
│   │   └── systems/            # ECS system implementations
│   │       ├── movement.hpp/cpp
│   │       ├── sensor.hpp/cpp
│   │       ├── combat.hpp/cpp
│   │       ├── energy.hpp/cpp
│   │       └── food_respawn.hpp/cpp
│   ├── evolution/              # NEAT: genome, neural network, NetworkCache, speciation
│   ├── visualization/          # SFML rendering (queries ECS directly)
│   ├── data/                   # CSV/JSON logger, metrics collector
│   └── gpu/                    # CUDA kernels, GpuDataBuffer, ECS-to-GPU packing
├── tests/                      # Google Test unit tests
├── analysis/                   # Python simulation analysis package and generated report output
├── profiler/                   # Python profiler analysis package and generated report output
├── docs/                       # Project documents (PDFs + LLD LaTeX source)
├── web/                        # GitHub Pages website
└── .github/workflows/          # CI/CD pipelines
```

### Simulation Output

Each run writes to `output/{experiment_name}/` (named experiments) or `output/YYYYMMDD_HHMMSS_seedN/` (anonymous runs):

| File | Contents |
|------|----------|
| `config.json` | Full config snapshot for this run |
| `stats.csv` | One row per report window: `step, predator_count, prey_count, births, deaths, num_species, avg_complexity, avg_predator_energy, avg_prey_energy` |
| `species.csv` | One row per species per generation |
| `genomes.json` | Representative genome snapshots (nodes + connections JSON) |

## C++ Code Style

MoonAI follows the **LLVM coding style** (2-space indentation, LLVM brace breaking, etc.).

### Code Quality Tools

Code formatting and static analysis are available via just commands:

| Command | Purpose |
|---------|---------|
| `just lint` | Auto-format all C++ files and run cppcheck static analysis |

**Run manually before committing** — these are not automatic during build.

### Style Configuration

- **`.clang-format`** — LLVM-based configuration in project root
  - 2-space indentation
  - 80 column limit
  - Attached braces
  - Right-aligned pointers/references

### Code Style Conventions

| Convention | Rule |
|------------|------|
| Namespace | `moonai` (CUDA internals: `moonai::gpu`) |
| Include paths | Relative to `src/`: `#include "core/types.hpp"` |
| Header guards | `#pragma once` |
| Member variables | Trailing underscore: `speed_`, `position_` |
| Functions / variables | `snake_case` |
| Classes / structs | `PascalCase` |

## Project

### Team

- Caner Aras
- Emir Irkılata
- Oğuzhan Özkaya

### Supervisor
- Ayşenur Birtürk

### Jury Members
- Deniz Canturk
- Mehmet Evren Coskun

*This project is developed as part of the CMPE 491/492 Senior Design course at TED University.*
