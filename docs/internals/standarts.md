---
description: Conventions, rules and policies for MoonAI development.
---

# Standards

## Rust

### Toolchain

- **Rust**: 1.95.0
- **MSRV**: 1.95.0
- **Edition**: 2024
- **Resolver**: 2

### Code Style

- Max line width: 120 characters
- Unix newlines
- Imports and modules auto-sorted
- `use_field_init_shorthand`, `use_try_shorthand` enabled

### Lints

#### Rust Lints

- **deny**: `elided_lifetimes_in_paths`, `absolute_paths_not_starting_with_crate`
- **warn**: `unsafe_code`, `unused`

#### Clippy Lint Groups (All denied)

- `correctness`, `suspicious`, `complexity`, `perf`, `style`

#### Clippy Individual Lints (All Denied)

- `dbg_macro`, `expect_used`, `unwrap_used`, `panic`, `todo`
- `needless_collect`, `redundant_clone`, `large_stack_arrays`
- `missing_const_for_fn`, `option_if_let_else`
- `print_stdout`, `print_stderr`
- `clone_on_ref_ptr`, `rest_pat_in_fully_bound_structs`, `str_to_string`

#### Clippy Thresholds

- `too-many-arguments-threshold`: 12
- `cognitive-complexity-threshold`: 15
- `enum-variant-size-threshold`: 128
- `type-complexity-threshold`: 256

### Rules

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

### Source Organization

- The Rust rewrite uses a single root package declared in `Cargo.toml`
- Core modules live directly under `src/` as flat files
- Only `src/ui/` and `src/tick/` may be subdirectories
- `src/ui/` owns UI/runtime rendering code
- `src/tick/` owns merged simulation, evolution, and CUDA-facing code

### Dependencies

- Declare runtime dependencies in `[dependencies]`
- Declare build-only dependencies in `[build-dependencies]`
- Keep dependency declarations version-pinned in the root manifest
- Do not add intra-project path dependencies; share code through `crate::...` modules instead

## Python

### Toolchain

- **uv**: 0.11+
- **Python**: 3.14+

#### Code Style

- Quote style: double
- Indent style: space
