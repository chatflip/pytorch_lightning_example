SRC_DIR := src
TEST_DIR := tests

.PHONY: install lint format test

.DEFAULT_GOAL := help

help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  install - Install dependencies"
	@echo "  format - Format code using ruff"
	@echo "  lint - Lint code using ruff and ty"
	@echo "  test - Run tests"

install:
	@uv sync --all-groups
	@pre-commit install

format:
	@uv run --frozen ruff check --fix --unsafe-fixes ${SRC_DIR}
	@uv run --frozen ruff format ${SRC_DIR}

lint:
	@uv run --frozen ruff check ${SRC_DIR}
	@uv run --frozen ty check ${SRC_DIR}

test:
	@uv run --frozen --dev pytest ${TEST_DIR}
