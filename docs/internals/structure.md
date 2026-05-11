---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Map

```
moonai/
├── .github/                    # GitHub workflows
├── analysis/                   # Python simulation analysis package
├── assets/                     # Static assets
├── docs/                       # Documentation source
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

> **Legacy C++ Implementation**: The original C++ simulation code is preserved in `legacy/` (git ignored). This legacy codebase can be inspected for reference but is no longer actively developed. It includes the CMake build system, full SFML visualization, and all original NEAT implementation details. All C++ build configuration (CMakeLists.txt, CMakePresets.json, .clang-format, .clang-tidy, vcpkg.json), source code (main.cpp, app/, core/, evolution/, metrics/, simulation/, visualization/), and architecture documentation (architecture.md) are located in `legacy/`.

## Rust Source Layout (`src/`)

MoonAI uses a single crate with a flattened source tree. Only `ui/` and `tick/` are subdirectories.

```
src/
├── lib.rs                      # Shared crate root for runtime code and ABI generation
├── main.rs                     # Binary entry point and UI bootstrap
├── experiment.rs               # Experiment catalog loading, SimulationConfig, defaults, and validation
├── settings.rs                 # settings.json loading/saving + AppSettings + UiConfig
├── metrics.rs                  # Metrics logger facade
├── profiler.rs                 # Runtime scope profiler tree and formatting helpers
├── ui/
│   ├── app.rs                  # Top-level UI shell, tabs, queue orchestration, and settings editor
│   ├── run_queue.rs            # Queued run snapshots and run history tracking
│   ├── session.rs              # One active simulation session, logging, overlays, and in-run controls
│   ├── render.rs               # Camera math and neural-network panel drawing
│   ├── types.rs                # UI runtime state, history structs, and selection types
│   └── world.rs                # Custom egui_wgpu world renderer and selected-agent overlays
└── tick/                       # Merged simulation + evolution runtime
```

`src/ui/world.rs` owns the custom `egui_wgpu` callback renderer used for instanced world drawing. `src/ui/app.rs` owns the application shell and queue flow, while `src/ui/session.rs` owns one live run at a time.

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
