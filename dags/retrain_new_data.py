from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

import os

from docker.types import Mount

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.sensors.python import PythonSensor
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator


# ---------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------
NEW_DATA_DIR = Path("/data/new_train_data")
ARCHIVE_DIR = Path("/data/new_train_data_processed")
OUTPUT_CSV = Path("/data/raw/pseudo_labeled_samples.csv")

MIN_SAMPLES = 200

TRAINING_IMAGE = "rakuten-text-trainer:latest"


def get_mlflow_env():
    return {
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        "MLFLOW_TRACKING_USERNAME": os.getenv("MLFLOW_TRACKING_USERNAME"),
        "MLFLOW_TRACKING_PASSWORD": os.getenv("MLFLOW_TRACKING_PASSWORD"),
    }


DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def has_enough_samples() -> bool:
    if not NEW_DATA_DIR.exists():
        return False
    return len(list(NEW_DATA_DIR.glob("*.json"))) >= MIN_SAMPLES


def jsons_to_csv():
    if not NEW_DATA_DIR.exists():
        raise AirflowSkipException("Kein new_train_data-Verzeichnis")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(NEW_DATA_DIR.glob("*.json"))[:MIN_SAMPLES]

    if len(json_files) < MIN_SAMPLES:
        raise AirflowSkipException("Nicht genug JSON-Samples")

    rows = []
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_run_dir = ARCHIVE_DIR / now
    archive_run_dir.mkdir(parents=True, exist_ok=True)

    for fp in json_files:
        with fp.open("r", encoding="utf-8") as f:
            sample = json.load(f)

        rows.append(
            {
                "designation": sample["designation"],
                "description": sample.get("description", ""),
                "prdtypecode": sample["label"],
                "source": sample.get("source", "pseudo-label"),
                "timestamp": sample["timestamp"],
            }
        )

        fp.rename(archive_run_dir / fp.name)

    df = pd.DataFrame(rows).drop_duplicates()

    tmp = OUTPUT_CSV.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(OUTPUT_CSV)

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


# ---------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------
with DAG(
    dag_id="retrain_text_classifier",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule="*/10 * * * *",  # alle 10 Minuten
    catchup=False,
    max_active_runs=1,
    tags=["rakuten"],
):
    wait_for_samples = PythonSensor(
        task_id="wait_for_enough_new_samples",
        python_callable=has_enough_samples,
        poke_interval=60,
        timeout=60 * 60 * 12,  # max 12h warten
        mode="reschedule",
    )

    convert_jsons = PythonOperator(
        task_id="convert_jsons_to_csv",
        python_callable=jsons_to_csv,
    )

    retrain_model = DockerOperator(
        task_id="retrain_model",
        image=TRAINING_IMAGE,
        docker_url="unix://var/run/docker.sock",
        mounts=[
            Mount(
                source="/data",
                target="/data",
                type="bind",
                read_only=False,
            )
        ],
        mount_tmp_dir=False,
        environment=get_mlflow_env(),
        auto_remove=True,
    )

    wait_for_samples >> convert_jsons >> retrain_model
