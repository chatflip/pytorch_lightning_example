#!/bin/bash
set -eu

uv run python src/train.py \
  -c config/experiments/food101_efficientnet_b0.yaml\
  --validate

# uv run python src/validate.py \
#  -c config/experiments/food101_efficientnet_b0.yaml\
#  --run-id fd61ec658f9d46519480c8a4b8519e4b
