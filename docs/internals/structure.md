---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Structure

```
moonai/
├── CMakeLists.txt              # Root CMake configuration
├── CMakePresets.json           # Build presets for Linux/Windows
├── vcpkg.json                  # Dependency manifest
├── pyproject.toml              # Python package config (hatchling build)
├── uv.lock                     # Python dependency lock
├── justfile                    # Project commands
├── config.lua                  # Unified config: default run + experiment matrix
├── .clang-format               # LLVM code style configuration
├── .clang-tidy                 # Static analysis configuration
├── src/
│   ├── main.cpp                # Entry point: CLI parsing and app startup
│   ├── profiler_main.cpp       # Profiler executable entry point
│   ├── app/                    # Application orchestration layer
│   ├── core/                   # Foundation code: types, config, Lua runtime, RNG
│   ├── metrics/                # Metrics aggregation and CSV/JSON logging
│   ├── simulation/             # ECS-based simulation core
│   ├── evolution/              # NEAT evolution implementation
│   └── visualization/          # SFML rendering and UI
├── tests/                      # Google Test unit tests
├── analysis/                   # Python simulation analysis package
├── profiler/                   # Python profiler analysis package
├── docs/                       # Documentation
│   ├── web/                    # GitHub Pages website
│   └── reports/                # Project reports
└── .github/workflows/          # CI/CD pipelines
```
