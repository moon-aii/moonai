---
description: Conventions, rules and policies for MoonAI development.
---

# Development

## Standards

### Rust

#### Toolchain

- **Rust**: 1.95.0
- **MSRV**: 1.95.0
- **Edition**: 2024
- **Resolver**: 2

#### Code Style

- Max line width: 120 characters
- Unix newlines
- Imports and modules auto-sorted
- `use_field_init_shorthand`, `use_try_shorthand` enabled

#### Lints

**Rust Lints:**

- **deny**: `elided_lifetimes_in_paths`, `absolute_paths_not_starting_with_crate`
- **warn**: `unsafe_code`, `unused`

**Clippy Lint Groups (All denied):**

- `correctness`, `suspicious`, `complexity`, `perf`, `style`

**Clippy Individual Lints (All Denied):**

- `dbg_macro`, `expect_used`, `unwrap_used`, `panic`, `todo`
- `needless_collect`, `redundant_clone`, `large_stack_arrays`
- `missing_const_for_fn`, `option_if_let_else`
- `print_stdout`, `print_stderr`
- `clone_on_ref_ptr`, `rest_pat_in_fully_bound_structs`, `str_to_string`

**Clippy Thresholds:**

- `too-many-arguments-threshold`: 12
- `cognitive-complexity-threshold`: 15
- `enum-variant-size-threshold`: 128
- `type-complexity-threshold`: 256

#### Rules

- **Suppression comments are forbidden**
  - Do NOT use: `#[allow(...)]`, `#![allow(...)]`, `#[expect(...)]`
  - Enforced by ripgrep in the quality gate
- **`unwrap`/`expect` are forbidden**
  - only permitted in test functions via `clippy.toml` settings (`allow-expect-in-tests`, `allow-unwrap-in-tests`)
  - use `?`, `Option::ok()`, or `anyhow::Context`
  - Propagate errors with `?` — never swallow errors silently
- **No `panic!()`/`todo!()`/`dbg!()`**
- **No print to stdout/stderr** - use `tracing::info!`, `tracing::warn!`, etc.
- **Clone explicitly on smart pointers** — `Arc::clone(&x)` not `x.clone()` (`clone_on_ref_ptr`)
- **`.to_owned()` not `.to_string()`** on `&str` values (`str_to_string`)
- **No `..` in fully-bound struct patterns** — all fields must be named (`rest_pat_in_fully_bound_structs`)
- **Enums over strings**

#### Source Organization

- The Rust rewrite uses a single root package declared in `Cargo.toml`
- Core modules live directly under `src/` as flat files
- Only `src/ui/` and `src/sim/` may be subdirectories
- `src/ui/` owns UI/runtime rendering code
- `src/sim/` owns merged simulation, evolution, and CUDA-facing code

#### Dependencies

- Declare runtime dependencies in `[dependencies]`
- Declare build-only dependencies in `[build-dependencies]`
- Keep dependency declarations version-pinned in the root manifest
- Do not add intra-project path dependencies; share code through `crate::...` modules instead

### Python

#### Toolchain

- **uv**: 0.11+
- **Python**: 3.14+

#### Code Style

- Quote style: double
- Indent style: space

## Workflow

### `justfile` Commands

### Development Commands

```bash
just sync
just build-debug
just build
just run
just analyse
just docs
```

- `just sync` installs the Python environment used by analysis and docs tooling.
- `just build-debug` builds the debug binary and copies runtime assets into `target/debug/`.
- `just build` builds the release binary and copies runtime assets into `target/release/`.
- `just run` builds and launches the release runtime.
- `just analyse` converts completed run artifacts into a self-contained HTML report.
- `just docs` serves the documentation site locally through Zensical.

### Quality and Verification Commands

```bash
just check
just test
just ci
just qual
```

- `just check` runs formatting, lint, and repository policy checks.
- `just test` runs the automated Rust test suite.
- `just ci` is the main pre-integration quality gate and runs `check` followed by `test`.
- `just qual` runs fixers first and then reruns the full quality gate.

### Quality Gate

Before any commit or pull request:

```bash
just ci
```

This runs `just check` followed by `just test`.

### Documentation Deployment

Documentation is deployed through `.github/workflows/docs.yml`. On pushes to `main`, GitHub
Pages builds the site with Zensical and publishes the generated `site/` directory.
