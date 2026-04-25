from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from docker.types import Mount, DeviceRequest

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NEW_DATA_DIR = Path("/app/data/new_train_data")
ARCHIVE_DIR = Path("/app/data/new_train_data_archived")
OUTPUT_X_CSV = Path("/app/data/raw/X_train_new.csv")
OUTPUT_Y_CSV = Path("/app/data/raw/Y_train_new.csv")
LABEL_ENCODING_PATH = Path("/app/configs/label_encoding.json")

MIN_SAMPLES = 5000

DEVICE = os.getenv("DEVICE", "cpu")
USE_GPU = DEVICE != "cpu"
DOCKER_NAMESPACE = os.getenv("DOCKER_NAMESPACE", "mlops2026")
TRAINING_IMAGE = f"{DOCKER_NAMESPACE}/train-text:{DEVICE}"
EVALUATE_IMAGE = f"{DOCKER_NAMESPACE}/evaluate-text:{DEVICE}"
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def get_training_env():
    """Returns essential environment variables for the training container."""
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
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def has_enough_samples() -> bool:
    """Sensor check: Do we have at least MIN_SAMPLES JSON files?"""
    if not NEW_DATA_DIR.exists():
        return False
    return len(list(NEW_DATA_DIR.glob("*.json"))) >= MIN_SAMPLES


def jsons_to_csv():
    """Processes JSON files, archives them, and updates the training CSVs."""
    if not NEW_DATA_DIR.exists():
        raise AirflowSkipException("No new_train_data directory found")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_X_CSV.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(NEW_DATA_DIR.glob("*.json"))[:MIN_SAMPLES]

    if len(json_files) < MIN_SAMPLES:
        raise AirflowSkipException("Not enough JSON samples")

    with LABEL_ENCODING_PATH.open("r", encoding="utf-8") as f:
        idx_to_code = {int(k): v for k, v in json.load(f)["idx_to_code"].items()}

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_run_dir = ARCHIVE_DIR / now
    archive_run_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    for fp in json_files:
        with fp.open("r", encoding="utf-8") as f:
            all_data.append(json.load(f))
        shutil.copy2(fp, archive_run_dir / fp.name)
        fp.unlink()

    df = pd.DataFrame(all_data).drop_duplicates()

    df_x = df[["designation", "description"]].assign(productid="", imageid="")
    df_y = df[["label"]].rename(columns={"label": "prdtypecode"})
    df_y["prdtypecode"] = df_y["prdtypecode"].map(idx_to_code)

    df_x.to_csv(OUTPUT_X_CSV, index=True, encoding="utf-8")
    df_y.to_csv(OUTPUT_Y_CSV, index=True, encoding="utf-8")

    (archive_run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "row_count": len(df),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Success! Processed {len(df)} samples.")


def read_mlflow_run_id(**context):
    """Reads the MLflow run_id written by the training container and pushes it to XCom."""
    run_id_file = Path("/app/results/mlflow_run_id_finetune.txt")
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
    dag_id="finetune_text_classifier",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule="*/10 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["rakuten"],
) as dag:
    wait_for_samples = PythonSensor(
        task_id="wait_for_enough_new_samples",
        python_callable=has_enough_samples,
        poke_interval=60,
        timeout=60 * 60 * 12,
        mode="reschedule",
    )

    convert_jsons = PythonOperator(
        task_id="convert_jsons_to_csv",
        python_callable=jsons_to_csv,
    )

    run_pipeline = DockerOperator(
        task_id="run_finetune_with_dvc",
        image=TRAINING_IMAGE,
        docker_url="unix://var/run/docker.sock",
        force_pull=False,
        mount_tmp_dir=False,
        auto_remove=True,
        mounts=[
            Mount(source=PROJECT_ROOT, target="/app", type="bind"),
            Mount(
                source="/var/run/docker.sock",
                target="/var/run/docker.sock",
                type="bind",
            ),
        ],
        working_dir="/app",
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])]
        if USE_GPU
        else None,
        command=[
            "/bin/bash",
            "-lc",
            """
            set -euo pipefail
            cd /app

            # 1. Install DVC
            uv tool install "dvc[s3]"
            export PATH="/home/appuser/.local/bin:$PATH"

            # 2. Sync dependencies
            export UV_LINK_MODE=copy
            rm -rf .venv
            uv sync --no-dev --frozen --extra training-text

            # 3. Track new data
            dvc add data/raw/X_train_new.csv data/raw/Y_train_new.csv
            git add data/raw.dvc .gitignore

            # 4. Run the entrypoint
            bash docker/entrypoint_finetune.sh
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
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])]
        if USE_GPU
        else None,
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

    promote_model_task = DockerOperator(
        task_id="promote_model_to_champion",
        image=EVALUATE_IMAGE,
        docker_url="unix://var/run/docker.sock",
        force_pull=False,
        mount_tmp_dir=False,
        auto_remove=True,
        entrypoint="python",
        command=[
            "-m",
            "src.serving.promote_mlflow_model",
            "--model-name",
            "text-classifier",
            "--alias",
            "champion",
            "--min-improvement",
            "0.003",
            "--output-path",
            "artifacts/mlflow_promotion_manifest.json",
        ],
        mounts=[
            Mount(source=PROJECT_ROOT, target="/app", type="bind"),
        ],
        environment=get_training_env(),
    )

    (
        wait_for_samples
        >> convert_jsons
        >> run_pipeline
        >> read_run_id
        >> run_evaluate
        >> promote_model_task
    )
