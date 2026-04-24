---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Structure

```
moonai/
├── .github/                    # GitHub workflows
├── analysis/                   # Python simulation analysis package
├── assets/                     # assets (fonts, logo)
├── docs/                       # Documentation source
│   ├── _assets/
│   │   ├── reports/
│   │   ├── logo.svg
│   │   ├── extra.css
│   │   └── extra.js
│   ├── internals/              # Developer documentation
│   │   ├── architecture.md
│   │   ├── roadmap.md
│   │   ├── standarts.md
│   │   ├── structure.md
│   │   └── workflow.md
│   ├── reports.md
│   ├── usage.md
│   ├── about.md
│   ├── installation.md
│   └── index.md
├── profiler/                   # Python profiler analysis package
├── src/                        # C++ simulation core
│   ├── main.cpp                # Entry point: CLI parsing and app startup
│   ├── profiler_main.cpp       # Profiler executable entry point
│   ├── app/                    # Application orchestration layer
│   ├── core/                   # Types, config, Lua runtime, seeded RNG
│   ├── evolution/              # NEAT genome, neural network, speciation
│   ├── metrics/                # CSV/JSON logging, aggregation
│   ├── simulation/             # ECS-based simulation (agents, physics, grid)
│   └── visualization/          # SFML rendering, UI overlay
├── tests/                      # Google Test unit tests
├── .clang-format               # LLVM code style configuration
├── .clang-tidy                 # Static analysis configuration
├── .gitattributes              # Git attributes
├── .gitignore                  # Git ignore rules
├── CMakeLists.txt              # Root CMake configuration
├── CMakePresets.json           # Build presets for Linux/Windows
├── README.md                   # Project readme
├── config.lua                  # Unified config: default run + experiment matrix
├── justfile                    # Project commands
├── pyproject.toml              # Python package config (hatchling build)
├── uv.lock                     # Python dependency lock
├── vcpkg.json                  # Dependency manifest
└── zensical.toml               # Website configuration
```

## Source Code (`src/`)

The C++ simulation is organized into:

| Directory | Purpose |
|-----------|---------|
| `core/` | Types, config, Lua runtime, seeded RNG |
| `app/` | Application orchestration, main loop |
| `simulation/` | ECS-based simulation (agents, physics, spatial grid) |
| `evolution/` | NEAT genome, neural network, speciation |
| `metrics/` | CSV/JSON logging, aggregation |
| `visualization/` | SFML rendering, UI overlay |

## `analysis/`

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point (`uv run analysis`) |
| `pipeline.py` | Orchestrates the analysis run |
| `io.py` | Run discovery, CSV/JSON loading |
| `labels.py` | Groups runs into experiment conditions |
| `plots.py` | Generates embedded matplotlib figures |
| `genome.py` | Renders neural network topology diagrams |
| `summary.py` | Prepares summary statistics |
| `html_report.py` | Renders self-contained HTML document |
| `report.html` | Jinja2 HTML report template |

## `profiler/`

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point (`uv run profiler`) |
| `report.py` | Generates profiler HTML report |
| `io.py` | Profile run discovery and validation |
| `html_report.py` | Renders self-contained HTML document |
| `report.html` | Jinja2 HTML report template |

## Documentation (`docs/`)

| Path | Purpose |
|------|---------|
| `index.md` | Documentation home |
| `usage.md` | Usage guide and CLI reference |
| `about.md` | Project overview and motivation |
| `installation.md` | Build and installation instructions |
| `reports.md` | Links to project reports |
| `internals/architecture.md` | System architecture |
| `internals/structure.md` | This file |
| `internals/workflow.md` | Development workflow |
| `internals/roadmap.md` | Tasks, bugs, and roadmap |
| `internals/standarts.md` | Coding standards |
