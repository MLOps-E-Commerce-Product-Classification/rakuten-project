#!/bin/bash
set -e

git config --global user.email "you@example.com"
git config --global user.name "DVC Runner"
git config --global --add safe.directory /app

echo ">>> Pulling data from DVC remote..."
uv run dvc pull data/raw/

echo ">>> Running DVC Pipeline..."
uv run dvc repro

echo ">>> Pushing results to DVC remote..."
uv run dvc push

echo ">>> Committing dvc.lock to Git..."
git add dvc.lock results/dvc_metrics.json

if git diff --cached --quiet; then
    echo "Nothing to commit, skipping."
else
    COMMIT_MSG="ci: dvc repro [$(git rev-parse --short HEAD)] - $(date '+%Y-%m-%d %H:%M')"
    git commit -m "$COMMIT_MSG"
    
    NEW_COMMIT_HASH=$(git rev-parse HEAD)
    RUN_ID_FILE="results/mlflow_run_id.txt"

    if [ -f "$RUN_ID_FILE" ]; then
        echo ">>> Logging final commit hash $NEW_COMMIT_HASH to MLflow..."
        RUN_ID=$(cat "$RUN_ID_FILE")
        
        uv run python - <<EOF
import mlflow
import os

with mlflow.start_run(run_id="$RUN_ID"):
    mlflow.log_param("final_pipeline_commit", "$NEW_COMMIT_HASH")
EOF
        rm "$RUN_ID_FILE"
    fi

    echo ">>> Pushing to Git remote..."
    git remote set-url origin https://${GIT_TOKEN}@dagshub.com/Mlops2026/rakuten-project.git
    git push
fi
