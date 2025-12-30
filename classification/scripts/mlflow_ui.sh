#!/bin/bash
set -eu

default_tracking_uri="sqlite:///mlflow.db"

uv run mlflow ui \
  --backend-store-uri $default_tracking_uri
