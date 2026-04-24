# Installation

## Pre-compiled binaries

To be added.

## Build from source

### Prerequisites

| Tool | Version | Required |
|------|---------|----------|
| C++ Compiler | C++17 support (GCC 9+, Clang 10+, MSVC 2019+) | Yes |
| CMake | 3.21+ | Yes |
| vcpkg | latest | Yes |
| CUDA Toolkit | 11.0+ | Yes |
| just | any | Recommended |
| uv | 0.11+ | For analysis only |

#### Just

[Just](https://github.com/casey/just) is a handy way to save and run project specific commands. Commands, called recipes, are stored in a file called `justfile` with syntax inspired by `make`. Recipes can be run with `just RECIPE`, and listed with `just --list`.

All of the commands needed for this project can be found and used from `justfile`. Despite being highly recommended, since Just is just a command wrapper it is not required to make this project work. Contents of the `justfile` can be used manually to standardize the commands.

```bash
# these are same
just clean
rm -rf build/

# clean recipe looks like this at the justfile
clean:
  rm -rf build/
```

### Clone the project

```bash
git clone https://github.com/moon-aii/moonai.git
cd moonai
```

### Simulation

#### 1. Configure

```bash
just configure
cmake --preset linux-debug # manually
```

#### 2. Build

```bash
just build
cmake --build build/linux-debug --parallel # manually
```

| Command | Description |
|---------|-------------|
| `just build` | Debug build |
| `just release` | Optimized release build |

##### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `MOONAI_BUILD_TESTS` | `ON` | Build unit tests |
| `MOONAI_BUILD_PROFILER` | `OFF` | Build profiler executable |

#### 3. Run

```bash
just run
```

### Analysis
