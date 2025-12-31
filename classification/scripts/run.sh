#!/bin/bash
set -eu

uv run python src/train.py \
  --config config/experiments/food101_efficientnet_b4.yaml\
  --validate

# uv run python src/train.py \
#   --config config/experiments/food101_efficientnet_b0.yaml\
#   --validate

# uv run python src/validate.py \
#  --config config/experiments/food101_efficientnet_b0.yaml\
#  --run-id de3b313b2f3640e4a7d6adf7c873cbae
