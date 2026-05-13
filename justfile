# MoonAI - Rust Project Commands
# Usage: just <recipe>
# Run `just --list` to see all available recipes.


# Set up Python environment
[group('build')]
sync:
  uv sync

# Build in debug mode
[group('build')]
build-debug:
  cargo build
  cp -r runtime/* target/debug

# Build in release mode
[group('build')]
build:
  cargo build --release
  cp -r runtime/* target/release


# Run the release build with bundled runtime files
[default]
[group('run')]
run: build
  target/release/moonai

# Run the debug build with bundled runtime files
[group('run')]
run-debug: build-debug
  target/build/moonai

# Generate the self-contained HTML analysis report from output/
[group('run')]
analyse:
  uv run analysis


# Fix: format and lint
[group('quality')]
fix:
  bunx prettier --log-level=warn --write .
  uv run ruff format .
  uv run ruff check . --fix
  cargo fmt --all
  cargo clippy --all-targets --all-features --fix --allow-dirty

# Check code: format, lint checks and manual supression command grep
[group('quality')]
check:
  bunx prettier --log-level warn --check .
  uv run ruff format . --check
  uv run ruff check .
  ! rg -n -F -e '#[allow' -e '#![allow' -g '*.rs' -g '!tests/**'
  cargo fmt --all -- --check
  cargo clippy --all-targets --all-features

# Run tests
[group('quality')]
test *args:
  cargo test --all-targets --all-features --locked -- --nocapture {{args}}

# Full check + test gate (github ci runs this command)
[group('quality')]
ci: check test

# Fix + Gate, prefer this recipe to save time instead of doing gate -> fix -> gate.
[group('quality')]
qual: fix ci

# Update dependencies
[group('dev')]
update:
  cargo update


# Remove build artifacts
[group('clean')]
clean:
  cargo clean
  uv run ruff clean
  rm -rf node_modules/
  rm -rf site/

# Remove all output and generated report artifacts
[group('clean')]
clean-outputs:
  rm -rf output/


# Clean and start docs website at localhost
[group('docs')]
docs:
  rm -rf site/
  uv run --group docs zensical serve

upgrade-check:
  cargo upgrade -i --dry-run

upgrade:
  cargo upgrade -i
