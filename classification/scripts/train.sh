#!/bin/bash
set -eu

uv run python src/train.py -c config/experiments/food101_efficientnet_b0.yaml
#uv run python src/validate.py -c config/experiments/food101_efficientnet_b0.yaml
