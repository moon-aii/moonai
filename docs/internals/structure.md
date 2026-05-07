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
├── build.rs                    # CUDA build script
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

## Rust Source Layout (`src/`)

MoonAI now uses a single crate with a flattened source tree. Only `ui/` and `tick/` are subdirectories.

```
src/
├── main.rs                     # Binary entry point and CLI routing
├── cli.rs                      # Clap args
├── config.rs                   # SimulationConfig defaults + serde
├── config_error.rs             # ConfigError + validation rules
├── lua.rs                      # Lua loading and moonai_defaults injection
├── settings.rs                 # settings.json loading + UiConfig
├── types.rs                    # Core shared types/constants
├── metrics.rs                  # Metrics logger facade
├── signal.rs                   # SIGINT/SIGTERM handling
├── ui/                         # UI runtime/rendering modules
└── tick/                       # Merged simulation + evolution runtime
```

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
