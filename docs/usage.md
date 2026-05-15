# Usage

## Run

```bash
just run
```

The application always launches the UI. Experiment selection, queueing, run replacement, and settings changes all happen inside the application.

MoonAI's main deliverable is the simulation environment itself. The runtime, queueing model,
exports, and analysis tooling are designed to help study evolutionary machine learning behavior,
not to present one fixed model checkpoint as the final outcome.

## Typical Workflow

1. Build and launch the application with `just run`.
2. Select a preset from `experiments.lua` in the Experiments tab.
3. Edit the draft configuration if needed.
4. Start the draft immediately or queue it for later execution.
5. Observe the live run in the Run tab and inspect individual agents if needed.
6. Review exported artifacts under `output/experiments/` after the run.
7. Generate a cross-run HTML report with `just analyse`.

## Runtime Files

MoonAI separates experiment definitions from persisted application settings:

- **`experiments.lua`** — experiment presets and simulation parameters
- **`settings.json`** — persisted application settings

Both files ship next to the binary. The runtime resolves them from `$(dirname $0)/`.

`settings.json` stores persisted application settings as a top-level object. UI-related values live under `ui`.
Color settings use hex strings: `#RRGGBB` for RGB and `#RRGGBBAA` for RGBA.

### `experiments.lua`

`experiments.lua` returns a named table of experiments. Each entry resolves to one full
`SimulationConfig` preset. The runtime injects `moonai_defaults`, so the file only needs to
override the parameters that differ from defaults.

```lua
-- moonai_defaults is injected by the runtime
local function extend(t, overrides) ... end

local conditions = {
    baseline = moonai_defaults,
    scale_5k = extend(moonai_defaults, {
        predator_count = 1250,
        prey_count = 3750,
    }),
}

local experiments = {}
for name, cfg in pairs(conditions) do
    experiments[name] = cfg
end

experiments["default"] = moonai_defaults
return experiments
```

The shipped file currently defines a broad experiment matrix rather than a single baseline only.
It includes 55 named condition groups with 5 fixed seeds each, plus a `default` entry for ad hoc
use. Presets can be selected, edited, queued, and run from the UI, but the Lua file itself is not
hot-reloaded during the session.

Set `seed` to `0` for random seed, or a fixed value for reproducible runs.

### `settings.json`

`settings.json` stores application settings with UI values nested under `ui`:

```json
{
  "ui": {
    "predator_color": "#FF6B36",
    "left_panel_width": 360.0,
    "right_panel_width": 360.0,
    "font_size": 12.0
  }
}
```

UI settings can be changed live from the Settings tab and saved back to disk without restarting the application.

## UI Flow

### Experiments

- select a preset loaded from `experiments.lua`
- edit the draft simulation parameters
- start the draft immediately or append it to the run queue
- replace the currently running session without restarting the application

### Queue

- pending runs execute sequentially
- when the active run finishes, the next queued run starts automatically
- queued runs store resolved config snapshots, so later editor changes do not rewrite already queued work
- a run with `max_ticks = 0` does not finish automatically and will block the queue until it is stopped or replaced

### Run

- shows the live simulation, overlays, agent inspection, profiler, and charts
- the active run continues even if another UI tab is open

### Settings

- edits persisted UI settings from `settings.json`
- applies changes live without restarting the application
- saves the updated settings file on demand

## Visualization Controls

| Key                    | Action                                               |
| ---------------------- | ---------------------------------------------------- |
| `Space`                | Pause / resume                                       |
| `↑` / `↓` or `+` / `-` | Increase / decrease simulation speed                 |
| `.`                    | Step one tick (while paused)                         |
| `Esc`                  | Quit                                                 |
| Left-click             | Select an agent and start camera follow              |
| Middle-click drag      | Pan camera and stop follow                           |
| Right-click drag       | Pan camera and stop follow                           |
| Scroll wheel           | Zoom                                                 |
| Home                   | Reset camera to default zoom and center; stop follow |

The main scene always renders the full active population: all predators, prey, and food with current positions, plus predator/prey movement directions and population overlay statistics.

The left overlay also includes a runtime profiler tree. It shows cumulative host-side scope timings as `frame -> child scopes`, with two-space indentation per nesting level, percent of total frame time, and average microseconds per simulation tick.

Visualization speed is separate from report export cadence:

- `speed_multiplier` controls UI refresh cadence only
- `1x` means the UI refreshes every tick
- `8x` means the UI refreshes every 8 ticks
- `report_interval_ticks` in `experiments.lua` controls when `stats.csv`, `species.csv`, `genomes.json`, and related artifacts are written

When an agent is selected, the camera automatically follows it until follow is disabled, the selection is cleared, or the agent disappears. Its **vision range** (semi-transparent circle), **sensor lines** (connections to nearby agents and food), and **stats panel** are automatically displayed on top of the normal full-population view. The agent controller receives 35 inputs: the 5 closest predators, prey, and food items as signed proximity-weighted `dx, dy` pairs, plus self energy, velocity `x/y`, and signed wall proximity on `x/y`. Missing targets are encoded as `0`, and closer objects produce larger absolute values in `[-1, 1]`. The **Network panel** shows its neural network topology with edges colored by weight value: blue (positive) -> gray (near zero) -> orange (negative).

## Analysis

Install analysis dependencies:

```bash
just sync
```

Generate a self-contained HTML report from `output/`:

```bash
just analyse
```

Internally this runs:

```bash
uv run analysis
```

The analysis writes a timestamped report to `output/analysis/`, for example `report_20260324_154233.html`.

## Simulation Output

Each run writes to `output/experiments/{experiment_name}_{unix_seconds}_seedN/`:

| File           | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.json`  | Full simulation config snapshot for this run                                                                                                                                                                                                                                                                                                                                                                                            |
| `stats.csv`    | One row per report interval sample, independent of visualization speed, with current state plus cumulative event totals: `tick, predator_count, prey_count, predator_births, prey_births, predator_deaths, prey_deaths, predator_species, prey_species, avg_predator_complexity, avg_prey_complexity, avg_predator_energy, avg_prey_energy, max_predator_generation, avg_predator_generation, max_prey_generation, avg_prey_generation` |
| `species.csv`  | One row per species per generation: `tick, population, species_id, size, avg_complexity`                                                                                                                                                                                                                                                                                                                                                |
| `genomes.json` | Representative genome snapshots (nodes + connections JSON)                                                                                                                                                                                                                                                                                                                                                                              |

## Output Artifacts

Generated artifacts live under `output/` (gitignored):

```
output/
├── experiments/           # Simulation run outputs
│   └── {experiment_name}_{unix_seconds}_seedN/
│       ├── config.json
│       ├── stats.csv
│       ├── species.csv
│       └── genomes.json
└── analysis/              # Analysis HTML reports
    └── report_*.html
```
