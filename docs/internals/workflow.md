---
description: Guide to set up, build, test, and deploy the project.
---

# Workflow

## `justfile` Commands

## Development Commands

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

## Quality and Verification Commands

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

## Quality Gate

Before any commit or pull request:

```bash
just ci
```

This runs `just check` followed by `just test`.

## Documentation Deployment

Documentation is deployed through `.github/workflows/docs.yml`. On pushes to `main`, GitHub
Pages builds the site with Zensical and publishes the generated `site/` directory.
