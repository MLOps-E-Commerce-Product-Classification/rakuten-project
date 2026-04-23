from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount, DeviceRequest

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEVICE = os.getenv("DEVICE", "cpu")
USE_GPU = DEVICE != "cpu"
DOCKER_NAMESPACE = os.getenv("DOCKER_NAMESPACE", "mlops2026")
TRAINING_IMAGE = f"{DOCKER_NAMESPACE}/train-text:{DEVICE}"
EVALUATE_IMAGE = f"{DOCKER_NAMESPACE}/evaluate-text:{DEVICE}"

# PROJECT_ROOT must be an absolute path on the host machine
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def get_training_env():
    """Extracts essential environment variables for the training container."""
    keys = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
        "DEVICE",
    ]
    return {k: os.getenv(k) for k in keys if os.getenv(k)}


DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def read_mlflow_run_id(**context):
    """Reads the MLflow run_id written by the training container and pushes it to XCom."""
    run_id_file = Path("/app/results/mlflow_run_id.txt")

    if not run_id_file.exists():
        raise FileNotFoundError(f"run_id file not found: {run_id_file}")

    run_id = run_id_file.read_text(encoding="utf-8").strip()

    if not run_id:
        raise ValueError("run_id file is empty")

    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
    print(f"MLflow run_id: {run_id}")
    return run_id


# ---------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------
with DAG(
    dag_id="manual_train_text",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1,
    tags=["rakuten", "manual"],
) as dag:
    dvc_track_data = DockerOperator(
        task_id="dvc_track_new_data",
        image=TRAINING_IMAGE,
        docker_url="unix://var/run/docker.sock",
        force_pull=False,
        mount_tmp_dir=False,
        auto_remove=True,
        mounts=[
            Mount(source=PROJECT_ROOT, target="/app", type="bind"),
        ],
        working_dir="/app",
        command=[
            "/bin/bash",
            "-lc",
            """
            set -e

            # 1. Ensure DVC is installed and available
            uv tool install "dvc[s3]"
            export PATH="/home/appuser/.local/bin:$PATH"

            # 2. Update DVC metadata for the tracked raw directory
            dvc add data/raw/X_train_update.csv data/raw/Y_train_CVw08PX.csv

            # 3. Stage the directory-level DVC metadata (data/raw is tracked as a whole)
            git add data/raw.dvc .gitignore

            # 4. Check if data actually changed (info only, training always runs)
            if git diff --cached --quiet; then
                echo "INFO: No changes detected in data/raw.dvc. Training will run anyway."
            else
                git commit -m "data: update raw directory - $(date '+%Y-%m-%d %H:%M')" || true
                git push || true

                # 5. Push data objects to the DVC remote (S3)
                dvc push
                echo "DVC push complete."
            fi
            """,
        ],
        environment={
            **get_training_env(),
            "GIT_TOKEN": os.getenv("GIT_TOKEN"),
            "GIT_AUTHOR_NAME": "Airflow MLOps",
            "GIT_AUTHOR_EMAIL": "mlops@rakuten.com",
            "GIT_COMMITTER_NAME": "Airflow MLOps",
            "GIT_COMMITTER_EMAIL": "mlops@rakuten.com",
            "DEVICE": DEVICE,
        },
    )

    run_train = DockerOperator(
        task_id="run_train_text",
        image=TRAINING_IMAGE,
        docker_url="unix://var/run/docker.sock",
        force_pull=False,
        mount_tmp_dir=False,
        auto_remove=True,
        shm_size="4G",
        mounts=[
            Mount(source=PROJECT_ROOT, target="/app", type="bind"),
            Mount(
                source="/var/run/docker.sock",
                target="/var/run/docker.sock",
                type="bind",
            ),
        ],
        working_dir="/app",
        device_requests=(
            [DeviceRequest(count=-1, capabilities=[["gpu"]])] if USE_GPU else None
        ),
        command=[
            "/bin/bash",
            "-lc",
            """
            set -e

            # 1. Ensure DVC is installed and available
            uv tool install "dvc[s3]"
            export PATH="/home/appuser/.local/bin:$PATH"

            # 2. Sync project dependencies (e.g. torch, transformers)
            export UV_LINK_MODE=copy
            rm -rf .venv
            uv sync --no-dev --frozen --extra training-text

            # 3. Stage changes like in the local Makefile
            git add src/ configs/ || true
            git commit -m "exp: manual airflow run - $(date '+%Y-%m-%d %H:%M')" || true

            # 4. Execute the training pipeline
            bash docker/entrypoint.sh
            """,
        ],
        environment={
            **get_training_env(),
            "GIT_TOKEN": os.getenv("GIT_TOKEN"),
            "GIT_AUTHOR_NAME": "Airflow MLOps",
            "GIT_AUTHOR_EMAIL": "mlops@rakuten.com",
            "GIT_COMMITTER_NAME": "Airflow MLOps",
            "GIT_COMMITTER_EMAIL": "mlops@rakuten.com",
            "DEVICE": DEVICE,
        },
    )

    read_run_id = PythonOperator(
        task_id="read_mlflow_run_id",
        python_callable=read_mlflow_run_id,
    )

    run_evaluate = DockerOperator(
        task_id="run_evaluate_text",
        image=EVALUATE_IMAGE,
        docker_url="unix://var/run/docker.sock",
        force_pull=False,
        mount_tmp_dir=False,
        auto_remove=True,
        mounts=[
            Mount(source=PROJECT_ROOT, target="/app", type="bind"),
        ],
        working_dir="/app",
        device_requests=(
            [DeviceRequest(count=-1, capabilities=[["gpu"]])] if USE_GPU else None
        ),
        # Entrypoint is already set in Dockerfile, just pass the arguments
        command=[
            "--mlflow_run_id",
            "{{ ti.xcom_pull(task_ids='read_mlflow_run_id', key='mlflow_run_id') }}",
            "--x_data_csv_path",
            "data/processed/val.csv",
            "--y_data_csv_path",
            "data/processed/val.csv",
            "--model_weights_path",
            "models/best_text_model.pt",
            "--label_encoding_path",
            "configs/label_encoding.json",
        ],
        environment={
            **get_training_env(),
            "DEVICE": DEVICE,
        },
    )

    dvc_track_data >> run_train >> read_run_id >> run_evaluate
