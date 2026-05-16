---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Map

```
moonai/
├── .github/                    # GitHub workflows (currently docs deployment)
├── analysis/                   # Python simulation analysis package
├── assets/                     # Static assets
├── docs/                       # Documentation source
├── papers/                     # Course reports and LaTeX sources
├── runtime/                    # Shipped runtime assets (experiments.lua, settings.json, assets/)
├── src/                        # Single-crate Rust source tree
├── .gitattributes              # Git attributes
├── .gitignore                  # Git ignore rules
├── build.rs                    # CUDA build script + shared ABI header generation
├── Cargo.toml                  # Rust package manifest
├── Cargo.lock                  # Locked dependency versions
├── clippy.toml                 # Clippy linter configuration
├── rustfmt.toml                # Rust formatter configuration
├── rust-toolchain.toml         # Rust toolchain specification
├── ruff.toml                   # Ruff linter configuration for Python
├── README.md                   # Project readme
├── justfile                    # Rust project commands
├── pyproject.toml              # Python package config
├── uv.lock                     # Python dependency lock
└── zensical.toml               # Website configuration
```

Generated outputs are written under `output/` at runtime and are gitignored. The most important
subdirectories are `output/experiments/` for run artifacts and `output/analysis/` for HTML
analysis reports.

## Rust Source Layout (`src/`)

MoonAI uses a single crate with a flattened source tree. Only `ui/` and `sim/` are subdirectories.

```
src/
├── lib.rs                      # Shared crate root for runtime code and ABI generation
├── main.rs                     # Binary entry point and UI bootstrap
├── experiment.rs               # Experiment catalog loading, SimulationConfig, defaults, and validation
├── settings.rs                 # settings.json loading/saving + AppSettings + UiConfig
├── metrics.rs                  # Metrics logger facade
├── profiler.rs                 # Runtime scope profiler tree and formatting helpers
├── sim/
│   ├── mod.rs                  # Public sim API, shared ABI/readback types, and module wiring
│   ├── helpers.rs              # Rust-side generic CUDA helper bindings
│   ├── helpers.cu              # CUDA malloc/memcpy/memset helper implementations
│   ├── bootstrap.rs            # Rust-side startup/device allocation orchestration + bootstrap FFI
│   ├── bootstrap.cu            # Population seeding, food seeding, counters, and free-list bootstrap kernels
│   ├── spatial_grid.rs         # Rust-side spatial-grid orchestration + grid FFI
│   ├── spatial_grid.cu         # Grid counting, scan finalization, and scatter kernels
│   ├── sensing.rs              # Rust-side sensor compute/snapshot entrypoints
│   ├── sensing.cu              # Sensor-input and sensor-snapshot kernels
│   ├── dynamics.rs             # Rust-side inference, vitals, movement, and tick entrypoints
│   ├── dynamics.cu             # Inference, vitals, movement, and tick kernels
│   ├── food.rs                 # Rust-side food resolution orchestration + FFI
│   ├── food.cu                 # Food claim, finalize, and respawn kernels
│   ├── combat.rs               # Rust-side combat resolution orchestration + FFI
│   ├── combat.cu               # Predator-prey combat claim/finalize kernels
│   ├── reproduction.rs         # Rust-side reproduction orchestration, expansion, and batched GPU dispatch
│   ├── reproduction.cu         # Reproduction reset, pair search, free-list, and parent-energy kernels
│   ├── metrics.rs              # Rust-side live-count, UI stats, and report-refresh entrypoints
│   ├── metrics.cu              # Metrics reduction, UI stats, and free-list-state kernels
│   ├── render.rs               # Rust-side render snapshot readback entrypoints
│   ├── render.cu               # Render snapshot packing kernels
│   ├── network_compilation.rs  # Rust-side compile/species/network readback entrypoints
│   ├── network_compilation.cu  # GPU network compilation and species classification kernels
│   ├── crossover.rs            # Rust-side crossover FFI owner
│   ├── crossover.cu            # GPU crossover logic
│   ├── mutation.rs             # Rust-side mutation FFI owner
│   ├── mutation.cu             # GPU mutation logic
│   └── sim.cuh                 # Minimal shared CUDA-only constants and helper utilities
├── ui/
│   ├── app.rs                  # Top-level UI shell, tabs, queue orchestration, and settings editor
│   ├── run_queue.rs            # Queued run snapshots and run history tracking
│   ├── session.rs              # One active simulation session, logging, overlays, and in-run controls
│   ├── render.rs               # Camera math and neural-network panel drawing
│   ├── types.rs                # UI runtime state, history structs, and selection types
│   └── world.rs                # Custom egui_wgpu world renderer and selected-agent overlays
```

`src/ui/world.rs` owns the custom `egui_wgpu` callback renderer used for instanced world drawing.
`src/ui/app.rs` owns the application shell and queue flow, while `src/ui/session.rs` owns one
live run at a time. `src/sim/mod.rs` is the main bridge between the Rust runtime and the CUDA
implementation compiled through `build.rs`.

## Delivery and Deployment Files

| Path                         | Purpose                                            |
| ---------------------------- | -------------------------------------------------- |
| `runtime/experiments.lua`    | Shipped experiment matrix and defaults overrides   |
| `runtime/settings.json`      | Shipped persisted UI settings                      |
| `.github/workflows/docs.yml` | GitHub Pages deployment for the documentation site |
| `papers/`                    | Course report sources and generated report PDFs    |

## `analysis/`

| File             | Purpose                                  |
| ---------------- | ---------------------------------------- |
| `__main__.py`    | CLI entry point (`uv run analysis`)      |
| `pipeline.py`    | Orchestrates the analysis run            |
| `io.py`          | Run discovery, CSV/JSON loading          |
| `labels.py`      | Groups runs into experiment conditions   |
| `plots.py`       | Generates embedded matplotlib figures    |
| `genome.py`      | Renders neural network topology diagrams |
| `summary.py`     | Prepares summary statistics              |
| `html_report.py` | Renders self-contained HTML document     |
| `report.html`    | Jinja2 HTML report template              |

The analysis package consumes completed runs from `output/experiments/`, groups them by
condition, and writes HTML reports to `output/analysis/`.

## Documentation (`docs/`)

| Path                     | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `_assets`                | Documentation assets, extra.css, extra.js, and reports |
| `index.md`               | Documentation home                                     |
| `usage.md`               | Usage guide and UI workflow                            |
| `about.md`               | Project overview and motivation                        |
| `installation.md`        | Build and installation instructions                    |
| `reports.md`             | Links to project reports                               |
| `internals/roadmap.md`   | Tasks, bugs, and roadmap                               |
| `internals/structure.md` | This file                                              |
| `internals/workflow.md`  | Development workflow                                   |
| `internals/standarts.md` | Coding standards                                       |
