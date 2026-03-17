# MoonAI

A modular and extensible simulation platform for studying evolutionary algorithms and neural network evolution through predator-prey dynamics.

**CMPE 491/492 - Senior Design Project | TED University**

🌐 **Website:** https://moon-aii.github.io/moonai/

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
- **Configurable** - All parameters adjustable via JSON without recompilation
- **Data Export** - CSV/JSON output (including optional per-tick trajectories) compatible with Python analysis tools

## Architecture

The system follows a modular architecture with four primary subsystems, each built as an independent static library:

```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization (SFML)                      │
│              Renders agents, grid, UI overlays               │
└──────────────────────────┬──────────────────────────────────┘
                           │ Observes State
┌──────────────────────────┴──────────────────────────────────┐
│                    Simulation Engine                         │
│         Physics loop, agent management, environment          │
└─────────┬──────────────────────────────────┬────────────────┘
          │ Queries Actions (GPU)            │ Exports Metrics
┌─────────┴────────────────┐    ┌────────────┴────────────────┐
│    Evolution Core (NEAT)  │    │     Data Management         │
│ Genome, NN, Species,     │    │  Logger (CSV), Metrics,     │
│ Mutation, Crossover       │    │  Config (JSON)              │
└──────────────────────────┘    └─────────────────────────────┘
```

| Subsystem | Library | Description |
|-----------|---------|-------------|
| `src/core/` | `moonai_core` | Shared types (`Vec2`), config loader, seeded RNG |
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
| `just run-config <path>` | Run with a custom config JSON |

### Visualization Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `↑` / `↓` or `+` / `-` | Increase / decrease simulation speed |
| `.` | Step one tick (while paused) |
| `H` | Toggle fast-forward mode (skip rendering for current generation) |
| `G` | Toggle grid overlay |
| `V` | Toggle vision range / sensor lines for selected agent |
| `R` | Reset simulation |
| `S` | Save screenshot |
| `Esc` | Quit |
| Left-click | Select an agent (shows stats + live NN panel) |
| Right-click drag | Pan camera |
| Scroll wheel | Zoom |

When an agent is selected, the **Network panel** (top-right) shows its topology with nodes colored by live activation value: blue (inactive, −1) → gray (zero) → orange (active, +1).

## Configuration

All experiment parameters are defined in `config/default_config.json` (40+ fields — snippet shows key ones):

```json
{
    "grid_width": 800,
    "grid_height": 600,
    "boundary_mode": "wrap",
    "predator_count": 50,
    "prey_count": 150,
    "vision_range": 100.0,
    "food_pickup_range": 10.0,
    "max_hidden_nodes": 50,
    "mutation_rate": 0.3,
    "add_node_rate": 0.03,
    "add_connection_rate": 0.05,
    "compatibility_threshold": 3.0,
    "complexity_penalty_weight": 0.01,
    "generation_ticks": 500,
    "seed": 0,
    "tick_log_enabled": false,
    "tick_log_interval": 10
}
```

Set `seed` to `0` for random seed, or a fixed value for reproducible experiments.

### Per-Tick Logging

Enable `tick_log_enabled: true` to write `ticks.csv` alongside the usual outputs. Every `tick_log_interval` ticks, one row per agent is appended:

```
generation,tick,agent_id,type,alive,x,y,energy,kills,food_eaten
```

Writes are buffered (flush every 500 rows) to minimise I/O overhead.

Run with a custom config:
```bash
just run-config path/to/my_config.json
```

## Running Experiments

### Quick start (full pipeline)

```bash
just experiment-pipeline
```

### Step by step

**1. Set up Python environment**
```bash
just setup-python        # installs pandas, matplotlib, networkx via uv
```

**2. Build release binary**
```bash
just release             # cmake --preset linux-release && cmake --build
```

**3. Create experiment configs**
```bash
just setup-experiments   # writes config/experiments/*.json
```

**4. Validate configs**
```bash
just validate-configs    # exits 1 if any parameter is out of range
```

**5. Run experiments**
```bash
just run-experiments     # 8 conditions × 5 seeds × 200 generations → output/
```

**6. Generate plots and summary**
```bash
just report              # reads output/, writes output/plots/*.png + summary.md
```

### Individual analysis scripts

| Script | Usage |
|--------|-------|
| `utils.py` | **Shared library** — `CONDITIONS`, `condition_label()`, `load_stats()`, `load_config()`, `find_runs()` |
| `report.py` | **Main post-run entry point** — all plots + summary table → `output/plots/` |
| `plot_fitness.py` | Fitness + complexity curve for one run (CLI + importable `plot()`) |
| `plot_population.py` | Predator/prey counts for one run (CLI + importable `plot()`) |
| `plot_species.py` | Species count + distribution for one run (CLI + importable `plot()`) |
| `plot_complexity.py` | Genome complexity for one run (CLI + importable `plot()`) |
| `compare_experiments.py` | Metric overlay for multiple runs (CLI + importable `compare()`) |
| `analyze_genome.py` | Network topology of best genome (uses networkx) |
| `run_experiments.py` | Batch headless runner — dispatches binary with multiple seeds |
| `setup_experiments.py` | Create experiment config files for all 8 conditions |
| `validate_config.py` | Validate config JSON against known parameter bounds (exit 1 on failure) |

### Analysis script conventions

- **All shared logic lives in `utils.py`** — do not duplicate `condition_label()`, `load_stats()`, `find_runs()`, or `CONDITIONS` elsewhere
- **Individual plot scripts must stay importable** — each exposes a module-level `plot()` or `compare()` function; `report.py` calls these directly (no subprocess)
- `matplotlib.use('Agg')` must appear before any `import matplotlib.pyplot` in every script that plots
- Scripts are run from the project root, so `analysis/` is on `sys.path` — use `from utils import ...` directly

### Experiment conditions

8 conditions, each overrides one parameter from `config/default_config.json`:

| Condition | Override |
|-----------|----------|
| `baseline` | — (default config) |
| `mut_low` | `mutation_rate: 0.1` |
| `mut_high` | `mutation_rate: 0.5` |
| `pop_small` | `predator_count: 25, prey_count: 75` |
| `pop_large` | `predator_count: 100, prey_count: 300` |
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
├── config/
│   ├── default_config.json     # Default experiment parameters (42 fields)
│   └── experiments/            # Per-condition configs (generated by setup_experiments.py)
├── src/
│   ├── main.cpp                # Entry point: CLI parsing, init, main loop, shutdown
│   ├── core/                   # Shared types (Vec2, AgentId), config loader, seeded RNG
│   ├── simulation/             # Agent hierarchy, environment, physics, spatial grid
│   ├── evolution/              # NEAT: genome, neural network, species, mutation, crossover
│   ├── visualization/          # SFML rendering (always compiled in; window suppressed by --headless)
│   ├── data/                   # CSV/JSON logger, metrics collector
│   └── gpu/                    # CUDA kernels (auto-detected; disabled at runtime by --no-gpu)
├── tests/                      # Google Test unit tests
├── analysis/                   # Python analysis scripts
├── documents/                  # Project documents (PDFs + LLD LaTeX source)
├── web/                        # GitHub Pages website
└── .github/workflows/          # CI/CD pipelines
```

### Simulation Output

Each run writes to `output/[YYYYMMDD_HHMMSS_seedN]/`:

| File | Contents |
|------|----------|
| `config.json` | Full config snapshot for this run |
| `stats.csv` | One row per generation: `generation, predator_count, prey_count, best_fitness, avg_fitness, num_species, avg_complexity` |
| `species.csv` | One row per species per generation |
| `genomes.json` | Best genome snapshots (nodes + connections JSON) |
| `ticks.csv` | Per-tick agent states (only when `tick_log_enabled: true`) |

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
