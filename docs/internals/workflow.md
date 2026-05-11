---
description: Guide to set up, build, test, and deploy the project.
---

# Workflow

## `justfile` Commands

## Development Commands

```bash
just build-debug
just build
just run
just analyse
```

## Quality Gate

Before any commit or pull request:

```bash
just ci
```

This runs `just check` followed by `just test`.
