# Installation

## Pre-compiled binaries

Pre-compiled binaries are not currently published. Build from source for now.

## Build from source

### Prerequisites

| Tool           | Version / Requirement                   | Required              |
| -------------- | --------------------------------------- | --------------------- |
| Rust toolchain | 1.95.0                                  | Yes                   |
| Cargo          | matching Rust toolchain                 | Yes                   |
| C++ compiler   | usable by `nvcc` as host compiler       | Yes                   |
| CUDA Toolkit   | recent toolkit with `nvcc` and `cudart` | Yes                   |
| just           | any                                     | Recommended           |
| uv             | 0.11+                                   | For analysis and docs |
| Python         | 3.14+                                   | For analysis and docs |

#### Just

[Just](https://github.com/casey/just) is a handy way to save and run project specific commands. Commands, called recipes, are stored in a file called `justfile` with syntax inspired by `make`. Recipes can be run with `just RECIPE`, and listed with `just --list`.

All of the commands needed for this project can be found and used from `justfile`. Despite being highly recommended, since Just is just a command wrapper it is not required to make this project work. Contents of the `justfile` can be used manually to standardize the commands.

```bash
# these are same
just clean
cargo clean

# clean recipe looks like this at the justfile
clean:
  cargo clean
```

### Clone the project

```bash
git clone https://github.com/moon-aii/moonai.git
cd moonai
```

### Simulation

#### 1. Build

```bash
just build-debug

# manual equivalent
cargo build
cp -r runtime/* target/debug
```

Release build:

```bash
just build

# manual equivalent
cargo build --release
cp -r runtime/* target/release
```

| Command            | Description                       |
| ------------------ | --------------------------------- |
| `just build-debug` | Debug build with runtime assets   |
| `just build`       | Release build with runtime assets |

The project uses `build.rs` to generate the shared Rust/CUDA ABI header and compile the CUDA
sources under `src/sim/`. There is no CMake or vcpkg step in the current implementation.

#### 2. Run

```bash
just run

# manual equivalent after a release build
target/release/moonai
```

Both `experiments.lua` and `settings.json` ship with the binary and are resolved from the binary directory.

### Analysis

Install the Python environment:

```bash
just sync
```

Generate the self-contained analysis report from `output/`:

```bash
just analyse
```

### Verification

Run the current automated tests:

```bash
just test
```

Run the full quality gate used before integration work:

```bash
just ci
```

### Documentation Site

Serve the documentation site locally:

```bash
just docs
```

This runs Zensical through `uv` and serves the generated site locally.
