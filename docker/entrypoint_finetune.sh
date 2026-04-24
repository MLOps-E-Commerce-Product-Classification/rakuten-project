#!/bin/bash
set -e

# Check if we have SSH access or need a token
if ! ssh-add -l > /dev/null 2>&1 && [ -z "$GIT_TOKEN" ]; then
    echo "ERROR: No SSH key found in agent AND GIT_TOKEN is not set!" >&2
    exit 1
fi

# If a token is available, redirect Git to use HTTPS instead of SSH
if [ -n "$GIT_TOKEN" ]; then
    echo ">>> Using GIT_TOKEN for authentication..."
    git config --global url."https://${GIT_TOKEN}@github.com/".insteadOf "git@github.com:"
else
    echo ">>> No GIT_TOKEN found, falling back to SSH..."
    export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"
fi

echo ">>> Pulling data from DVC remote..."
uv run dvc pull data/raw/

echo ">>> Running DVC Finetuning Pipeline..."
uv run dvc repro preprocess-text-new
uv run dvc repro finetune-text

echo ">>> Committing dvc.lock to Git..."
git add dvc.lock results/

if git diff --cached --quiet; then
    echo "Nothing to commit, skipping."
else
    COMMIT_MSG="ci: finetune repro [$(git rev-parse --short HEAD)] - $(date '+%Y-%m-%d %H:%M')"
    git commit --no-verify -m "$COMMIT_MSG"

    NEW_COMMIT_HASH=$(git rev-parse HEAD)
    RUN_ID_FILE="results/mlflow_run_id_finetune.txt"

    if [ -f "$RUN_ID_FILE" ]; then
        echo ">>> Logging final commit hash to MLflow..."
        RUN_ID=$(cat "$RUN_ID_FILE")

        uv run python - <<EOF
import mlflow

with mlflow.start_run(run_id="$RUN_ID"):
    mlflow.log_param("final_pipeline_commit", "$NEW_COMMIT_HASH")
EOF
    fi

    echo ">>> Pushing to Git remote..."
    git push origin HEAD
fi

echo ">>> Pushing results to DVC remote..."
uv run dvc push

git update-index --really-refresh > /dev/null 2>&1 || true
git reset --mixed HEAD > /dev/null 2>&1 || true

echo ">>> Finetuning finished successfully."
