#!/bin/bash
set -e

export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

echo ">>> Pulling data from DVC remote..."
uv run dvc pull data/raw/

echo ">>> Running DVC Pipeline..."
uv run dvc repro

echo ">>> Committing dvc.lock to Git..."
git add dvc.lock results/

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

with mlflow.start_run(run_id="$RUN_ID"):
    mlflow.log_param("final_pipeline_commit", "$NEW_COMMIT_HASH")
EOF
    fi

    echo ">>> Pushing to Git remote..."
    git push
fi

echo ">>> Pushing results to DVC remote..."
uv run dvc push

# Clean up any file-system drift between container and host
git update-index --really-refresh > /dev/null 2>&1 || true
git reset --mixed HEAD > /dev/null 2>&1 || true

echo ">>> Finished successfully."
