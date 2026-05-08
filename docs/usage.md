# Usage

## Run

```bash
just run
```

## Configuration

MoonAI separates simulation configuration from UI configuration:

- **`config/config.lua`** — simulation parameters (population, evolution, energy system, etc.)
- **`config/settings.json`** — UI configuration (colors, sizes, panel layout, window settings)

Both files live in the `config/` subdirectory of the binary. The runtime resolves them from `$(dirname $0)/config/`, or you can specify explicit paths via CLI flags.

### `config.lua`

Simulation config uses a single **`config.lua`** file at the project root. It returns a named table of experiments — every entry is a fully-specified run. The runtime injects C++ struct defaults as the `moonai_defaults` global (24000 predators, 96000 prey on a 3600×3600 square world), so Lua only needs to override what it changes.

```lua
-- moonai_defaults is injected by the runtime (mirrors C++ SimulationConfig defaults)
-- Defaults: 500 predators, 1500 prey (2000 total), 3000×3000 square world, 1500 ticks per report window
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

A single-entry file auto-selects without `--experiment`. The `default` entry serves as the everyday run config.

Set `seed` to `0` for random seed, or a fixed value for reproducible experiments in `config.lua`.

### CLI flags

| Flag                  | Purpose                                          |
| --------------------- | ------------------------------------------------ |
| `-h, --help`          | Show CLI help                                    |
| `-n, --ticks <n>`     | Override max ticks (`0` = infinite)              |
| `--headless`          | Run without visualization                        |
| `-v, --verbose`       | Enable debug logging                             |
| `--experiment <name>` | Select one experiment by name                    |
| `--all`               | Run all experiments sequentially (headless only) |
| `--list`              | List experiment names and exit                   |
| `--name <name>`       | Override output directory name                   |
| `--validate`          | Load + validate config, print result, exit       |

## Running Simulation

### Examples

```bash
just run                                              # GUI with default config
just run -- --headless                                # Headless mode
just run -- --experiment mut_low_seed42 --headless    # One experiment
just run-release -- --all --headless                  # Full batch (release build)
```

### List available experiments

Shows all experiments in config.lua.

```bash
just list-experiments
```

### Run experiments

275 seeded runs + default entry → output/

```bash
just experiment-run
```

## Visualization Controls

| Key                    | Action                                        |
| ---------------------- | --------------------------------------------- |
| `Space`                | Pause / resume                                |
| `↑` / `↓` or `+` / `-` | Increase / decrease simulation speed          |
| `.`                    | Step one tick (while paused)                  |
| `S`                    | Save screenshot                               |
| `Esc`                  | Quit                                          |
| Left-click             | Select an agent (shows stats + live NN panel) |
| Middle-click drag      | Pan camera                                    |
| Right-click drag       | Pan camera                                    |
| Scroll wheel           | Zoom                                          |
| Home                   | Reset camera to default zoom and center       |

The main scene always renders the full active population: all predators, prey, and food with current positions, plus predator/prey movement directions and population overlay statistics.

Visualization speed is separate from report export cadence:

- `speed_multiplier` controls UI refresh cadence only.
- `1x` means the UI refreshes every tick.
- `8x` means the UI refreshes every 8 ticks.
- `report_interval_ticks` in `config.lua` controls when `stats.csv`, `species.csv`, `genomes.json`, and related artifacts are written.

When an agent is selected, its **vision range** (semi-transparent circle), **sensor lines** (connections to nearby agents and food), and **stats panel** are automatically displayed on top of the normal full-population view. The agent controller receives 35 inputs: the 5 closest predators, prey, and food items as signed proximity-weighted `dx, dy` pairs, plus self energy, velocity `x/y`, and signed wall proximity on `x/y`. Missing targets are encoded as `0`, and closer objects produce larger absolute values in `[-1, 1]`. The **Network panel** shows its neural network topology with edges colored by weight value: blue (positive) → gray (near zero) → orange (negative).

## Set up Python and generate analysis

Installs simulation + profiler analysis dependencies via uv.

```bash
just setup-python
```

Reads output/, writes a self-contained HTML report.

```bash
just experiment-analyse
```

## Full pipeline

Runs all experiments + generates report.

```bash
just experiment
```

## Simulation Output

Each run writes to `output/experiments/{experiment_name}/` (named experiments) or `output/experiments/YYYYMMDD_HHMMSS_seedN/` (anonymous runs):

| File           | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.json`  | Full config snapshot for this run                                                                                                                                                                                                                                                                                                                                                                                                       |
| `stats.csv`    | One row per report interval sample, independent of visualization speed, with current state plus cumulative event totals: `tick, predator_count, prey_count, predator_births, prey_births, predator_deaths, prey_deaths, predator_species, prey_species, avg_predator_complexity, avg_prey_complexity, avg_predator_energy, avg_prey_energy, max_predator_generation, avg_predator_generation, max_prey_generation, avg_prey_generation` |
| `species.csv`  | One row per species per generation: `tick, population, species_id, size, avg_complexity`                                                                                                                                                                                                                                                                                                                                                |
| `genomes.json` | Representative genome snapshots (nodes + connections JSON)                                                                                                                                                                                                                                                                                                                                                                              |

## Analysis

The Python analysis tool generates self-contained HTML report for all qualifying runs in `output/`.

```bash
just experiment-analyse
```

Internally this runs the packaged analysis entry point from `analysis/`:

```bash
uv run analysis
```

The analysis writes a timestamped report to `output/analysis/`, for example `report_20260324_154233.html`.

The generated HTML is fully self-contained: it embeds all plots and report data directly into a single file, including:

- per-condition plots for population, species, complexity, and representative-genome topology
- cross-condition comparison plots using seed-aggregated statistics
- the grouped summary table at the final sampled generation
- skipped-run information for incomplete or invalid runs
- inline styling and navigation so the report opens directly in a browser without side files

## Output Artifacts

Generated artifacts live under `output/` (gitignored):

```
output/
├── experiments/           # Simulation run outputs
│   └── {experiment_name}/
│       ├── config.json
│       ├── stats.csv
│       ├── species.csv
│       └── genomes.json
├── profiler/              # Profiler outputs
│   ├── profiles/          # C++ profiler JSON data
│   │   └── YYYY-MM-DD_HH-MM-SS_*.json
│   └── profile.html       # Profiler HTML report
└── analysis/              # Analysis HTML reports
    └── report_*.html
```
