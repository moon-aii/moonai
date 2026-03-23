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
- **GPU Acceleration** - CUDA backend for batch neural inference — auto-enabled for populations >= 1000 agents, uses pinned host buffers, and falls back to the CPU path on runtime GPU failures
- **Cross-Platform** - Runs on Linux and Windows with identical behavior
- **Reproducible Experiments** - Seeded RNG with deterministic simulation for scientific rigor
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
| `src/gpu/` | `moonai_gpu` | CUDA kernels and batch runtime for neural inference (pinned host buffers + runtime CPU fallback) |

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

CUDA is enabled at runtime when available and the population is at least 1000 agents. If GPU upload or inference fails during execution, MoonAI disables the CUDA path and continues with CPU inference.

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

66 conditions defined in `config.lua` across 9 groups, each × 5 seeds = **330 deterministic runs**.

The default baseline is 2000 agents (500 predators, 1500 prey) on a 4300×2400 world with 1500 ticks/generation. Scaled experiments use `scale_base()` to maintain agent density by proportionally adjusting world size and food count. GPU is auto-enabled for populations >= 1000.

#### Group A — Baseline sweeps (2K agents)

| Condition | Override |
|-----------|----------|
| `baseline` | — (unmodified defaults) |
| `mut_low` | `mutation_rate: 0.1` |
| `mut_high` | `mutation_rate: 0.5` |
| `mut_very_low` | `mutation_rate: 0.05` |
| `mut_very_high` | `mutation_rate: 0.8` |
| `pop_small` | 100 pred + 300 prey (400 total, scaled world) |
| `pop_medium` | 250 pred + 750 prey (1K total, scaled world) |
| `pop_large` | 1250 pred + 3750 prey (5K total, scaled world) |
| `pop_huge` | 2500 pred + 7500 prey (10K total, scaled world) |
| `pop_massive` | 5000 pred + 15000 prey (20K total, scaled world) |
| `no_speciation` | `compatibility_threshold: 100` |
| `tight_speciation` | `compatibility_threshold: 1.0` |
| `tanh` | `activation_function: "tanh"` |
| `relu` | `activation_function: "relu"` |
| `crossover_low` | `crossover_rate: 0.25` |
| `crossover_none` | `crossover_rate: 0.0` |

#### Group B — Scale experiments (proportional world)

| Condition | Total Agents | World (approx) |
|-----------|-------------|-----------------|
| `scale_1k` | 1,000 | 3040×1700 |
| `scale_3k` | 3,000 | 5270×2940 |
| `scale_5k` | 5,000 | 6800×3800 |
| `scale_8k` | 8,000 | 8600×4800 |
| `scale_10k` | 10,000 | 9600×5400 |
| `scale_15k` | 15,000 | 11760×6615 |
| `scale_20k` | 20,000 | 13580×7640 |

#### Group C — Parameter sweeps at 5K

| Condition | Override |
|-----------|----------|
| `s5k_mut_low` | `mutation_rate: 0.1` |
| `s5k_mut_high` | `mutation_rate: 0.5` |
| `s5k_mut_very_high` | `mutation_rate: 0.8` |
| `s5k_tanh` | `activation_function: "tanh"` |
| `s5k_relu` | `activation_function: "relu"` |
| `s5k_no_spec` | `compatibility_threshold: 100` |
| `s5k_tight_spec` | `compatibility_threshold: 1.0` |
| `s5k_crossover_low` | `crossover_rate: 0.25` |
| `s5k_crossover_none` | `crossover_rate: 0.0` |

#### Group D — Parameter sweeps at 10K

| Condition | Override |
|-----------|----------|
| `s10k_mut_low` | `mutation_rate: 0.1` |
| `s10k_mut_high` | `mutation_rate: 0.5` |
| `s10k_tanh` | `activation_function: "tanh"` |
| `s10k_relu` | `activation_function: "relu"` |
| `s10k_no_spec` | `compatibility_threshold: 100` |
| `s10k_crossover_low` | `crossover_rate: 0.25` |

#### Group E — World density (5K agents, varying world size)

| Condition | World | Density |
|-----------|-------|---------|
| `dense_5k` | 3000×1700 | Very high — constant encounters |
| `normal_5k` | 6800×3800 | Proportional (same as scale_5k) |
| `sparse_5k` | 12000×6750 | Low — agents rarely meet |
| `vast_5k` | 15000×8400 | Extremely sparse |

#### Group F — Generation length

| Condition | Base | Ticks |
|-----------|------|-------|
| `ticks_500_2k` | 2K | 500 |
| `ticks_2000_2k` | 2K | 2000 |
| `ticks_3000_2k` | 2K | 3000 |
| `ticks_500_5k` | 5K | 500 |
| `ticks_2000_5k` | 5K | 2000 |
| `ticks_3000_5k` | 5K | 3000 |

#### Group G — Energy / resource dynamics

| Condition | Base | Override |
|-----------|------|----------|
| `energy_scarce_2k` | 2K | `initial_energy: 75, food_respawn_rate: 0.01` |
| `energy_abundant_2k` | 2K | `initial_energy: 300, food_respawn_rate: 0.05` |
| `energy_scarce_5k` | 5K | `initial_energy: 75, food_respawn_rate: 0.01` |
| `energy_abundant_5k` | 5K | `initial_energy: 300, food_respawn_rate: 0.05` |
| `energy_extreme_5k` | 5K | `initial_energy: 50, food_respawn_rate: 0.005, energy_drain: 0.15` |
| `energy_rich_5k` | 5K | `initial_energy: 500, food_respawn_rate: 0.08, energy_drain: 0.03` |

#### Group H — Agent speed / interaction range (5K)

| Condition | Override |
|-----------|----------|
| `fast_agents_5k` | `predator_speed: 6.0, prey_speed: 7.0` |
| `slow_agents_5k` | `predator_speed: 2.5, prey_speed: 3.0` |
| `wide_vision_5k` | `vision_range: 300` |
| `narrow_vision_5k` | `vision_range: 80` |
| `long_attack_5k` | `attack_range: 40` |
| `short_attack_5k` | `attack_range: 10` |

#### Group I — Topology complexity

| Condition | Base | Override |
|-----------|------|----------|
| `high_complexity_5k` | 5K | `add_node_rate: 0.1, add_connection_rate: 0.15` |
| `low_complexity_5k` | 5K | `add_node_rate: 0.01, add_connection_rate: 0.02` |
| `no_growth_5k` | 5K | `add_node_rate: 0.0, add_connection_rate: 0.0` |
| `high_complexity_10k` | 10K | `add_node_rate: 0.1, add_connection_rate: 0.15` |
| `max_hidden_small_5k` | 5K | `max_hidden_nodes: 10` |
| `max_hidden_large_5k` | 5K | `max_hidden_nodes: 50` |

### Large-scale experiments

Experiments with 5K+ agents require significant compute. Recommendations:

- **GPU strongly recommended** for populations >= 2000 (auto-enabled when CUDA is available)
- **Release build** (`just release`) for 2-5x faster simulation
- **Headless mode** (`--headless`) disables rendering for maximum throughput
- **Visual mode** can also use CUDA inference now, but headless mode still gives the best throughput because rendering stays on the CPU/SFML side
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

# Quick FPS benchmark in visual mode (requires display)
just bench-fps

# Profile with perf (Linux, requires perf installed)
just profile

# Build with AddressSanitizer + UBSan and run 5 headless generations
just check-memory

# Run GPU tests locally (requires CUDA)
just test-gpu
```

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
