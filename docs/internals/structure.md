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
├── src/                        # Single-crate Rust source tree
├── runtime/                    # Runtime assets (config/, assets/)
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

MoonAI now uses a single crate with a flattened source tree. Only `ui/` and `tick/` are subdirectories.

```
src/
├── lib.rs                      # Shared crate root for runtime code and ABI generation
├── main.rs                     # Binary entry point and CLI routing
├── config.rs                   # SimulationConfig loading, defaults, and validation
├── settings.rs                 # settings.json loading + UiConfig
├── metrics.rs                  # Metrics logger facade
├── ui/                         # UI runtime/rendering modules, egui panels, and custom wgpu world renderer
└── tick/                       # Merged simulation + evolution runtime
```

`src/ui/world.rs` owns the custom `egui_wgpu` callback renderer used for instanced world drawing. `src/ui/render.rs` remains focused on camera math, screenshots, and low-volume overlays instead of the hot-path scene draw.

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
| `usage.md`               | Usage guide and CLI reference                          |
| `about.md`               | Project overview and motivation                        |
| `installation.md`        | Build and installation instructions                    |
| `reports.md`             | Links to project reports                               |
| `internals/roadmap.md`   | Tasks, bugs, and roadmap                               |
| `internals/structure.md` | This file                                              |
| `internals/workflow.md`  | Development workflow                                   |
| `internals/standarts.md` | Coding standards                                       |
