#!/bin/bash
set -eu

uv run python src/train.py \
  --config config/experiments/food101_mobilenet_v2.yaml\
  --validate

uv run python src/train.py \
  --config config/experiments/food101_efficientnet_b4.yaml\
  --validate

# uv run python src/export_onnx.py \
#   --run-id 8b632d114a2a4379a36189f74d2f8de6
