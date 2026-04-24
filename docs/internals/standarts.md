---
description: Conventions, rules and policies.
---

# Standarts

## C++ Code Style

MoonAI follows the **LLVM coding style** (2-space indentation, LLVM brace breaking, etc.).

### Style Configuration

- **`.clang-format`** — LLVM-based configuration in project root
  - 2-space indentation
  - 120 column limit
  - Attached braces
  - Left-aligned pointers/references

### Code Style Conventions

| Convention | Rule |
|------------|------|
| Namespace | `moonai` |
| Include paths | Relative to `src/`: `#include "core/types.hpp"` |
| Header guards | `#pragma once` |
| Member variables | Trailing underscore: `speed_`, `position_` |
| Functions / variables | `snake_case` |
| Classes / structs | `PascalCase` |
