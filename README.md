# MoonAI

A modular and extensible simulation platform for studying evolutionary algorithms and neural network evolution through predator-prey dynamics.

**CMPE 491/492 - Senior Design Project | TED University**

**Website:** https://moon-aii.github.io/moonai/

## Overview

MoonAI uses a simplified predator-prey environment as a synthetic benchmark to evaluate evolutionary computation methods. Agents (predators and prey) are controlled by neural networks whose structure and weights evolve over generations using the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm.

The platform enables researchers to:

- Observe how neural network topologies emerge and grow in complexity through evolution
- Compare different genetic representations, mutation strategies, and selection methods
- Generate structured datasets for machine learning research without real-world data
- Visualize agent behavior and algorithm evolution in real time

### Key Features

- **NEAT Implementation** - Evolves both topology and weights of neural networks simultaneously
- **Real-Time Visualization** - SFML-based rendering with interactive controls and live NN activation display
- **GPU Acceleration** - CUDA backend for batch neural inference and fitness evaluation with runtime CPU fallback
- **Cross-Platform** - Runs on Linux and Windows with identical behavior
- **Reproducible Experiments** - Seeded RNG with deterministic simulation for scientific rigor
- **Configurable** - All parameters adjustable via Lua configs without recompilation
- **Data Export** - CSV/JSON output (including optional per-tick trajectories) compatible with Python analysis tools

## Architecture

The system follows a modular architecture with four primary subsystems, each built as an independent static library:

```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization (SFML)                     │
│              Renders agents, grid, UI overlays              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Observes State
┌──────────────────────────┴──────────────────────────────────┐
│                    Simulation Engine                        │
│         Physics loop, agent management, environment         │
└─────────┬──────────────────────────────────┬────────────────┘
          │ Queries Actions (GPU)            │ Exports Metrics
┌─────────┴────────────────┐    ┌────────────┴────────────────┐
│    Evolution Core (NEAT) │    │     Data Management         │
│ Genome, NN, Species,     │    │  Logger (CSV), Metrics,     │
│ Mutation, Crossover      │    │  Config (JSON)              │
└──────────────────────────┘    └─────────────────────────────┘
```

| Subsystem | Library | Description |
|-----------|---------|-------------|
| `src/core/` | `moonai_core` | Shared types (`Vec2`), Lua config loader (sol2), seeded RNG |
| `src/simulation/` | `moonai_simulation` | Agent hierarchy, environment grid, collision/sensing |
| `src/evolution/` | `moonai_evolution` | NEAT genome, neural network, speciation, mutation, crossover |
| `src/visualization/` | `moonai_visualization` | SFML window, renderer, UI overlay |
| `src/data/` | `moonai_data` | CSV logger, metrics collector |
| `src/gpu/` | `moonai_gpu` | CUDA kernels for batch inference and fitness evaluation |

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

Or with just:
```bash
just setup-vcpkg
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
just test
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

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `MOONAI_BUILD_TESTS` | `ON` | Build unit tests |

### Runtime Modes

Mode selection happens at runtime via flags — no need to rebuild:

| Command | Description |
|---------|-------------|
| `just run` | Default: visualization window, GPU if available |
| `just run-headless` | No window, max speed (auto-switches if `$DISPLAY` unset) |
| `just run-no-gpu` | Force CPU inference even if CUDA is compiled in |
| `just run-server` | Headless + CPU-only (for servers without a display or GPU) |
| `just run-config <path>` | Run with a custom config file |

### Visualization Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `↑` / `↓` or `+` / `-` | Increase / decrease simulation speed |
| `.` | Step one tick (while paused) |
| `H` | Toggle fast-forward mode (skip rendering for current generation) |
| `G` | Toggle grid overlay |
| `V` | Toggle vision range / sensor lines for selected agent |
| `E` | Open experiment selector (multi-config only) |
| `R` | Reset simulation |
| `S` | Save screenshot |
| `Esc` | Quit |
| Left-click | Select an agent (shows stats + live NN panel) |
| Right-click drag | Pan camera |
| Scroll wheel | Zoom |

When an agent is selected, the **Network panel** (top-right) shows its topology with nodes colored by live activation value: blue (inactive, −1) → gray (zero) → orange (active, +1).

## Configuration

Configuration uses a single **`config.lua`** file at the project root. It returns a named table of experiments — every entry is a fully-specified run. The runtime injects C++ struct defaults as the `moonai_defaults` global, so Lua only needs to override what it changes.

### `config.lua` structure

```lua
-- moonai_defaults is injected by the runtime (mirrors C++ SimulationConfig defaults)
local function extend(t, overrides) ... end

local conditions = {
    baseline = moonai_defaults,
    mut_low  = extend(moonai_defaults, { mutation_rate = 0.1 }),
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

A single-entry file auto-selects without `--experiment`. The `default` entry serves as the everyday run config.

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
./moonai config.lua --experiment default              # GUI with default config
./moonai config.lua                                   # GUI with experiment selector
./moonai config.lua --experiment mut_low_seed42 --headless  # One experiment
./moonai config.lua --all --headless                  # Full batch
./moonai config.lua --experiment default --set mutation_rate=0.1  # Ad-hoc override
```

Set `seed` to `0` for random seed, or a fixed value for reproducible experiments.

### Per-Tick Logging

Enable `tick_log_enabled = true` to write `ticks.csv` alongside the usual outputs. Every `tick_log_interval` ticks, one row per agent is appended:

```
generation,tick,agent_id,type,alive,x,y,energy,kills,food_eaten
```

Writes are buffered (flush every 500 rows) to minimise I/O overhead.

## Running Experiments

### Quick start (full pipeline)

```bash
just experiment-pipeline    # runs all experiments + generates report
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
just experiments            # 8 conditions × 5 seeds × 200 generations → output/
# or run a single experiment:
just run-experiment baseline_seed42
```

**4. Set up Python and generate report**
```bash
just setup-python           # installs pandas, matplotlib, networkx via uv
just report                 # reads output/, writes analysis/output/*.png + summary.md
```

### Analysis CLI (`analysis/cli.py`)

All analysis is accessed through a single entry point (`analysis/` contains its own `pyproject.toml`):

```bash
cd analysis && uv run python cli.py <command> [options]
```

| Command | Description |
|---------|-------------|
| `plot <run_dir>` | Fitness and complexity curves for one run |
| `population <run_dir>` | Predator/prey counts over generations |
| `species <run_dir>` | Species count and size distribution |
| `complexity <run_dir>` | Genome node/connection count over generations |
| `compare <run_dirs...>` | Overlay one metric across multiple runs (`--metric`, `--smooth`) |
| `genome <run_dir>` | Neural network topology of the best genome (`-g` for generation) |
| `report` | Generate complete report: all plots + summary table |

All commands accept `-o <path>` to save as PNG instead of displaying interactively. Run `cd analysis && uv run python cli.py <command> --help` for full options.

The library modules (`plot_fitness.py`, `plot_population.py`, `plot_species.py`, `plot_complexity.py`, `compare_experiments.py`, `analyze_genome.py`, `report.py`) expose their functions for import — all shared utilities live in `utils.py`.

### Experiment conditions

8 conditions defined in `config.lua`, each overrides one parameter from `moonai_defaults`:

| Condition | Override |
|-----------|----------|
| `baseline` | — (default config) |
| `mut_low` | `mutation_rate: 0.1` |
| `mut_high` | `mutation_rate: 0.5` |
| `pop_small` | `predator_count: 40, prey_count: 120` |
| `pop_large` | `predator_count: 200, prey_count: 600` |
| `no_speciation` | `compatibility_threshold: 100` |
| `tanh` | `activation_function: "tanh"` |
| `crossover_low` | `crossover_rate: 0.25` |

## Development

```bash
# Generate compile_commands.json for your IDE/LSP
just compdb

# Format code
just format

# Run static analysis (cppcheck)
just lint

# Benchmark NN forward-pass timing (requires release build + pop_large config)
just bench-nn

# Quick FPS benchmark in visual mode (requires display)
just bench-fps

# Profile with perf (Linux, requires perf installed)
just profile

# Build with AddressSanitizer + UBSan and run 5 headless generations
just check-memory
```

## Project Structure

```
moonai/
├── CMakeLists.txt              # Root CMake configuration
├── CMakePresets.json            # Build presets for Linux/Windows
├── vcpkg.json                  # Dependency manifest
├── justfile                    # Project commands (run `just --list` for full list)
├── config.lua                  # Unified config: default run + experiment matrix (8 × 5 seeds)
├── src/
│   ├── main.cpp                # Entry point: CLI parsing, init, main loop, shutdown
│   ├── core/                   # Shared types (Vec2, AgentId), config loader, seeded RNG
│   ├── simulation/             # Agent hierarchy, environment, physics, spatial grid
│   ├── evolution/              # NEAT: genome, neural network, species, mutation, crossover
│   ├── visualization/          # SFML rendering (always compiled in; window suppressed by --headless)
│   ├── data/                   # CSV/JSON logger, metrics collector
│   └── gpu/                    # CUDA kernels (auto-detected; disabled at runtime by --no-gpu)
├── tests/                      # Google Test unit tests
├── analysis/                   # Python analysis (self-contained: pyproject.toml + cli.py entry point)
├── docs/                       # Project documents (PDFs + LLD LaTeX source)
├── web/                        # GitHub Pages website
└── .github/workflows/          # CI/CD pipelines
```

### Simulation Output

Each run writes to `output/{experiment_name}/` (named experiments) or `output/YYYYMMDD_HHMMSS_seedN/` (anonymous runs):

| File | Contents |
|------|----------|
| `config.json` | Full config snapshot for this run |
| `stats.csv` | One row per generation: `generation, predator_count, prey_count, best_fitness, avg_fitness, num_species, avg_complexity` |
| `species.csv` | One row per species per generation |
| `genomes.json` | Best genome snapshots (nodes + connections JSON) |
| `ticks.csv` | Per-tick agent states (only when `tick_log_enabled: true`) |

### Project Documents

| Document | Description |
|----------|-------------|
| `docs/ProjectProposal.pdf` | Initial project proposal |
| `docs/ProjectSpecification.pdf` | Detailed project specifications |
| `docs/AnalysisReport.pdf` | Requirements analysis |
| `docs/HighLevelDesignReport.pdf` | System architecture and design |
| `docs/Poster.pdf` | Conference poster presentation |
| `docs/LowLevelDesignReport.pdf` | Detailed component design |

## C++ Code Style

| Convention | Rule |
|------------|------|
| Namespace | `moonai` (CUDA internals: `moonai::gpu`) |
| Include paths | Relative to `src/`: `#include "core/types.hpp"` |
| Header guards | `#pragma once` |
| Member variables | Trailing underscore: `speed_`, `position_` |
| Functions / variables | `snake_case` |
| Classes / structs | `PascalCase` |

## Team

| Name | Role |
|------|------|
| **Caner Aras** | Developer |
| **Emir Irkılata** | Developer |
| **Oğuzhan Özkaya** | Developer |

**Supervisor:** Ayşenur Birtürk
**Jury Members:** Deniz Canturk, Mehmet Evren Coskun

## License

This project is developed as part of the CMPE 491/492 Senior Design course at TED University.
