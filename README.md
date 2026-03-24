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
- **GPU Acceleration** - CUDA backend for GPU-resident sensing, inference, and headless tick processing at large populations, with runtime CPU fallback on GPU failures
- **Cross-Platform** - Runs on Linux and Windows with matched features and stable runtime behavior
- **Reproducible Experiments** - Seeded RNG with deterministic behavior within each execution backend; CPU and GPU runs are kept numerically close but are not bit-exact twins
- **Lua Scripting** - Config, custom fitness functions, and generation hooks — all in Lua without recompilation
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
| `src/core/` | `moonai_core` | Shared types (`Vec2`), Lua config loader (sol2), Lua runtime (fitness/hooks), seeded RNG |
| `src/simulation/` | `moonai_simulation` | Agent hierarchy, environment grid, collision/sensing |
| `src/evolution/` | `moonai_evolution` | NEAT genome, neural network, speciation, mutation, crossover |
| `src/visualization/` | `moonai_visualization` | SFML window, renderer, UI overlay |
| `src/data/` | `moonai_data` | CSV logger, metrics collector |
| `src/gpu/` | `moonai_gpu` | CUDA kernels and batch runtime for GPU-resident sensing, inference, and headless tick execution (with runtime CPU fallback) |

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
| `just run-headless` | No window, max speed (auto-switches if `$DISPLAY` unset) |
| `just run-no-gpu` | Force CPU inference even if CUDA is compiled in |
| `just run-server` | Headless + CPU-only (for servers without a display or GPU) |
| `just run-config <path>` | Run with a custom config file |

CUDA is enabled at runtime when available and the population is at least 1000 agents. In headless runs, the fast path keeps sensing, inference, and tick processing on the GPU for the whole generation. If GPU upload, sensing, inference, or resident tick execution fails during runtime, MoonAI disables the CUDA path and continues with CPU execution.

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

Configuration uses a single **`config.lua`** file at the project root. It returns a named table of experiments — every entry is a fully-specified run. The runtime injects C++ struct defaults as the `moonai_defaults` global (2000 agents on a 4300×2400 world), so Lua only needs to override what it changes.

### `config.lua` structure

```lua
-- moonai_defaults is injected by the runtime (mirrors C++ SimulationConfig defaults)
-- Defaults: 500 predators, 1500 prey (2000 total), 4300×2400 world, 1500 ticks/gen
local function extend(t, overrides) ... end

-- Helper: scale world and food proportionally to population
local function scale_base(pred, prey)
    local total = pred + prey
    local default_total = moonai_defaults.predator_count + moonai_defaults.prey_count
    local factor = math.sqrt(total / default_total)
    return {
        predator_count = pred, prey_count = prey,
        grid_width  = math.floor(moonai_defaults.grid_width * factor),
        grid_height = math.floor(moonai_defaults.grid_height * factor),
        food_count  = math.floor(moonai_defaults.food_count * (total / default_total)),
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

### Lua Callbacks

Experiments can optionally define Lua functions that the runtime calls at specific points. No callback defined means the default C++ behavior is used (zero overhead).

| Callback | Signature | Purpose |
|----------|-----------|---------|
| `fitness_fn` | `(stats, weights) -> number` | Custom fitness formula replacing the built-in linear combination |
| `on_generation_end` | `(gen, stats) -> table or nil` | Called after each generation; return a table of config overrides (e.g. `{ mutation_rate = 0.5 }`) or `nil` |
| `on_experiment_start` | `(config) -> nil` | Called once before the main loop |
| `on_experiment_end` | `(stats) -> nil` | Called once after the main loop |

Example with a custom fitness function and adaptive mutation hook:

```lua
experiments["adaptive"] = extend(moonai_defaults, {
    fitness_fn = function(stats, weights)
        return weights.survival * stats.age_ratio
             + weights.kill     * stats.kills_or_food
             + weights.energy   * stats.energy_ratio
             + stats.alive_bonus
             + weights.distance * stats.dist_ratio
             - weights.complexity_penalty * stats.complexity
    end,

    on_generation_end = function(gen, stats)
        if stats.avg_fitness < 2.0 and gen > 20 then
            return { mutation_rate = 0.5 }
        end
        return nil
    end,
})
```

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
just experiments            # 66 conditions × 5 seeds × 200 generations → output/
# or run a single experiment:
just run-experiment baseline_seed42
```

**4. Set up Python and generate analysis**
```bash
just setup-python           # installs simulation + profiler analysis dependencies via uv
just analyse                # reads output/, writes a self-contained HTML report
just analyse-profile        # reads output/profiles/, writes a profiler HTML report
```

### Analysis

The Python analysis tool has a single mode: it always generates one self-contained HTML report for all qualifying runs in `output/`.

```bash
just analyse
```

Internally this runs the packaged analysis entry point from `analysis/`:

```bash
cd analysis && uv run moonai-analysis
```

The analysis step is non-interactive and always writes a timestamped report to `analysis/output/`, for example `report_20260324_154233.html`.

The generated HTML is fully self-contained: it embeds all plots and report data directly into a single file, including:

- per-condition plots for fitness, population, species, complexity, and best-genome topology
- cross-condition comparison plots using seed-aggregated statistics
- the grouped summary table at the final sampled generation
- skipped-run information for incomplete or invalid runs
- inline styling and navigation so the report opens directly in a browser without side files

The analysis code is structured as a small package under `analysis/moonai_analysis/`:

- `pipeline.py` orchestrates the full analysis run
- `io.py` discovers runs and loads CSV/JSON data
- `labels.py` groups runs into experiment conditions
- `plots.py` generates embedded per-condition and comparison figures
- `genome.py` renders embedded best-genome topology diagrams
- `summary.py` prepares structured summary data for the report
- `html_report.py` renders the final self-contained HTML document
- `templates/report.html.j2` defines the HTML report layout

### Profiler output and analysis

Profiler runs now use the dedicated `moonai_profiler` entry point with a separate
`profiler.lua` config. The standard `moonai` binary no longer owns profiler
orchestration.

```bash
just profile
```

Each profiler suite writes to its own timestamped directory under `output/profiles/`
by default. Every suite contains six raw run artifacts plus one suite-level
aggregate artifact:

| File | Contents |
|------|----------|
| `raw/*/profile.json` | Full raw run payload: run metadata, event/counter definitions, per-generation records, and summary statistics |
| `profile_suite.json` | Suite manifest: six raw runs, dropped fastest/slowest runs, and aggregate timing/counter summaries from the remaining four runs |

The profiler suite uses six fixed seeds from `profiler.lua`, drops the fastest and
slowest runs by average generation time, and reports aggregate timing/counter data
from the remaining four runs. Standard simulation builds do not include profiler
instrumentation, so normal runtime overhead stays unchanged.

To generate the standalone profiler report:

```bash
just analyse-profile
```

Internally this runs the packaged profiler entry point via `just analyse-profile`.

The profiler writes a timestamped self-contained HTML report to `profiler/output/`, for example `profile_report_20260324_154233.html`.

The profiler package lives under `profiler/moonai_profiler/` and includes:

- `pipeline.py` for orchestration
- `io.py` for discovering and validating `profile_suite.json` runs
- `plots.py` for embedded timing charts
- `html_report.py` for rendering
- `templates/report.html.j2` for layout

### Experiment conditions

66 conditions defined in `config.lua` across 9 groups, each × 5 seeds = **330 deterministic runs**.

The default baseline is 2000 agents (500 predators, 1500 prey) on a 4300×2400 world with 1500 ticks/generation. Scaled experiments use `scale_base()` to maintain agent density by proportionally adjusting world size and food count. GPU is auto-enabled for populations >= 1000.

- Group A — Baseline sweeps (2K agents)
- Group B — Scale experiments (proportional world)
- Group C — Parameter sweeps at 5K
- Group D — Parameter sweeps at 10K
- Group E — World density (5K agents, varying world size)
- Group F — Generation length
- Group G — Energy / resource dynamics
- Group H — Agent speed / interaction range (5K)
- Group I — Topology complexity

### Large-scale experiments

Experiments with 5K+ agents require significant compute. Recommendations:

- **GPU strongly recommended** for populations >= 2000 (auto-enabled when CUDA is available)
- **Release build** (`just release`) for 2-5x faster simulation
- **Headless mode** (`--headless`) disables rendering for maximum throughput
- **Visual mode** can still use CUDA-assisted sensing/inference, but headless mode gives the best throughput because rendering stays on the CPU/SFML side and the full resident GPU path is only used there
- **Memory**: ~4 GB RAM for 10K agents, ~8 GB for 20K agents
- **VRAM**: ~512 MB for 10K agents, ~1 GB for 20K agents
- Running all 330 experiments sequentially takes significant time; use `--experiment` to run specific conditions or parallelize across machines

## Development

```bash
# Generate compile_commands.json for your IDE/LSP
just compdb

# Format code
just format

# Run static analysis (cppcheck)
just lint

# Benchmark NN forward-pass timing (requires release build)
just bench-nn

# Run the dedicated profiler suite entry point
just profile

# Generate the standalone profiler HTML report
just analyse-profile

# Run profiler and then build the profiler report
just profile-pipeline

# Quick FPS benchmark in visual mode (requires display)
just bench-fps

# Build with AddressSanitizer + UBSan and run 5 headless generations
just check-memory

# Run GPU tests locally (requires CUDA)
just test-gpu
```

The dedicated profiler writes one `profile_suite.json` file per suite under a unique
directory in `output/profiles/` by default when invoked through `just profile`.

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
│   ├── core/                   # Shared types (Vec2, AgentId), config loader, Lua runtime, seeded RNG
│   ├── simulation/             # Agent hierarchy, environment, physics, spatial grid
│   ├── evolution/              # NEAT: genome, neural network, species, mutation, crossover
│   ├── visualization/          # SFML rendering (always compiled in; window suppressed by --headless)
│   ├── data/                   # CSV/JSON logger, metrics collector
│   └── gpu/                    # CUDA kernels (auto-detected; disabled at runtime by --no-gpu)
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
