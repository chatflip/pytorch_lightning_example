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
- Sync: `make install`
- Python version: `uv python pin 3.12`

## Prohibited

- `uv pip install`
- `uv pip install <package>@latest`
- `pip install`
