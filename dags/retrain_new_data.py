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
from docker.types import Mount

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# Use /app because the Airflow container mounts the repo there
NEW_DATA_DIR = Path("/app/data/new_train_data")
ARCHIVE_DIR = Path("/app/data/new_train_data_processed")
OUTPUT_X_CSV = Path("/app/data/raw/X_train_new.csv")
OUTPUT_Y_CSV = Path("/app/data/raw/Y_train_new.csv")
LABEL_ENCODING_PATH = Path("/app/configs/label_encoding.json")

MIN_SAMPLES = 10

# Dynamic Image Selection based on .env / environment
DEVICE = os.getenv("DEVICE", "cpu")
TRAINING_IMAGE = f"rakuten-ml/train-text:{DEVICE}"
PROJECT_ROOT = os.getenv("PROJECT_ROOT")  # Needs to be absolute host path (from .env)


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

    # Load label encoding for idx -> prdtypecode mapping
    with LABEL_ENCODING_PATH.open("r", encoding="utf-8") as f:
        label_encoding = json.load(f)
    idx_to_code = {int(k): v for k, v in label_encoding["idx_to_code"].items()}

    # Create timestamped archive directory
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_run_dir = ARCHIVE_DIR / now
    archive_run_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    for fp in json_files:
        with fp.open("r", encoding="utf-8") as f:
            sample = json.load(f)
        all_data.append(sample)
        shutil.copy2(fp, archive_run_dir / fp.name)
        fp.unlink()  # Delete original after processing

    df = pd.DataFrame(all_data).drop_duplicates()

    # Build X: designation, description, productid, imageid
    df_x = df[["designation", "description"]].copy()
    df_x["productid"] = ""
    df_x["imageid"] = ""

    # Build Y: map label back to prdtypecode
    df_y = df[["label"]].copy()
    df_y.columns = ["prdtypecode"]
    df_y["prdtypecode"] = df_y["prdtypecode"].map(idx_to_code)

    # Overwrite the CSVs for retraining
    df_x.to_csv(OUTPUT_X_CSV, index=True, sep=",", encoding="utf-8")
    df_y.to_csv(OUTPUT_Y_CSV, index=True, sep=",", encoding="utf-8")

    # Save metadata for archival purposes
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


# ---------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------
with DAG(
    dag_id="retrain_text_classifier",
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
        command=[
            "/bin/bash",
            "-lc",
            """
            set -euo pipefail
            cd /app

            # 1. Install DVC
            uv tool install "dvc[s3]"
            export PATH="/home/appuser/.local/bin:$PATH"

            # 2. Sync mit training-text extra (= text + data)
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
            "GIT_AUTHOR_NAME": "Airflow MLOps",
            "GIT_AUTHOR_EMAIL": "mlops@rakuten.com",
            "GIT_COMMITTER_NAME": "Airflow MLOps",
            "GIT_COMMITTER_EMAIL": "mlops@rakuten.com",
            "DEVICE": DEVICE,
        },
    )

    wait_for_samples >> convert_jsons >> run_pipeline
