#!/bin/bash
# Run formatter and linter for Python files
set -e

echo "Running make format..."
make format

echo "Running make lint..."
make lint

echo "All Python checks passed!"

