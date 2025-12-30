---
description: "Package management and dependency installation using uv"
globs:
alwaysApply: false
---

# Package Management

Use `uv` exclusively. Never use `pip`, `poetry`, or `pipenv`.

## Commands

- Install: `uv add <package>`
- Dev dependency: `uv add --dev <package>`
- Sync: @scripts/install.sh

## Prohibited

- `uv pip install`
- `uv pip install <package>@latest`
- `pip install`
