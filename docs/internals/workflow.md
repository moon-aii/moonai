---
description: Guide to set up, build, test, and deploy the project.
---

# Workflow

## Development

### Commands

```bash
# Generate compile_commands.json for your IDE/LSP
just compdb

# Run tests
just test              # basic run
just test --verbose    # verbose output
just test -R GpuTest   # filter tests

# Code formatting and linting
just lint              # Auto-format and run static analysis
```
