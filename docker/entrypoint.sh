#!/bin/bash
set -e

echo ">>> Pulling data from DVC remote..."
uv run dvc pull data/raw/

echo ">>> Running DVC Pipeline..."
uv run dvc repro

echo ">>> Pushing results to DVC remote..."
uv run dvc push
