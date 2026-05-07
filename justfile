# MoonAI - Rust Project Commands
# Usage: just <recipe>
# Run `just --list` to see all available recipes.


# Set up Python environment
[group('build')]
sync:
  uv sync

# Build all crates in the workspace
[group('build')]
build-debug:
  cargo build --workspace
  cp -r runtime/* target/debug

# Build in release mode
[group('build')]
build:
  cargo build --workspace --release
  cp -r runtime/* target/release


# Run the release build with default config (pass additional args after --)
[default]
[group('run')]
run *args: build
  cargo run --release {{args}}

# Run the debug build with default config (pass additional args after --)
[group('run')]
run-debug *args: build-debug
  cargo run {{args}}

# Generate the self-contained HTML analysis report from output/
[group('run')]
analyse:
  uv run analysis


# Fix: format and lint
[group('quality')]
fix:
  uv run ruff format .
  uv run ruff check . --fix

  prettier --log-level=warn --write .

  cargo fmt --all
  cargo clippy --workspace --all-targets --all-features --fix --allow-dirty


# Check code: format, lint checks and manual supression command grep
[group('quality')]
check:
  uv run ruff format . --check
  uv run ruff check .

  prettier --log-level warn --check .

  ! rg -n -F -e '#[allow' -e '#![allow' -g '*.rs' -g '!tests/**'
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets --all-features

# Run tests
[group('quality')]
test *args:
  cargo test --workspace --all-targets --all-features --locked -- --nocapture {{args}}

# Full check + test gate (github ci runs this command)
[group('quality')]
gate: check test


# Update dependencies
[group('dev')]
update:
  cargo update


# Remove build artifacts
[group('clean')]
clean:
  cargo clean
  uv run ruff clean

# Remove all output and generated report artifacts
[group('clean')]
clean-outputs:
  rm -rf output/


# Clean and start docs website at localhost
[group('docs')]
docs:
  rm -rf site/
  uv run --group docs zensical serve
