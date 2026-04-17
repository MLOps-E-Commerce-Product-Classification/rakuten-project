from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import os
import shutil


from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.sensors.python import PythonSensor
from airflow.operators.python import PythonOperator


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NEW_DATA_DIR = Path("/app/data/new_train_data")
ARCHIVE_DIR = Path("/app/data/new_train_data_processed")
OUTPUT_X_CSV = Path("/app/data/raw/X_train_new.csv")
OUTPUT_Y_CSV = Path("/app/data/raw/Y_train_new.csv")
LABEL_ENCODING_PATH = Path("/app/configs/label_encoding.json")

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
        raise AirflowSkipException("No new_train_data directory found")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_X_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_Y_CSV.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(NEW_DATA_DIR.glob("*.json"))[:MIN_SAMPLES]

    if len(json_files) < MIN_SAMPLES:
        raise AirflowSkipException("Not enough JSON samples")

    # Load label encoding for idx -> prdtypecode mapping
    with LABEL_ENCODING_PATH.open("r", encoding="utf-8") as f:
        label_encoding = json.load(f)
    idx_to_code = {int(k): v for k, v in label_encoding["idx_to_code"].items()}

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_run_dir = ARCHIVE_DIR / now
    archive_run_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    for fp in json_files:
        with fp.open("r", encoding="utf-8") as f:
            sample = json.load(f)
        all_data.append(sample)
        shutil.copy2(fp, archive_run_dir / fp.name)
        fp.unlink()

    df = pd.DataFrame(all_data).drop_duplicates()

    # Build X: designation, description, productid, imageid
    df_x = df[["designation", "description"]].copy()
    df_x["productid"] = ""
    df_x["imageid"] = ""

    # Build Y: label -> prdtypecode, dann idx -> code mapping
    df_y = df[["label"]].copy()
    df_y.columns = ["prdtypecode"]
    df_y["prdtypecode"] = df_y["prdtypecode"].map(idx_to_code)

    # Write X and Y as CSV (with index)
    df_x.to_csv(OUTPUT_X_CSV, index=True, sep=",", encoding="utf-8")
    df_y.to_csv(OUTPUT_Y_CSV, index=True, sep=",", encoding="utf-8")

    print(f"Fertig! {len(df)} Zeilen wurden verarbeitet.")

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
    schedule="*/10 * * * *",  # every 10 minutes
    catchup=False,
    max_active_runs=1,
    tags=["rakuten"],
):
    wait_for_samples = PythonSensor(
        task_id="wait_for_enough_new_samples",
        python_callable=has_enough_samples,
        poke_interval=60,
        timeout=60 * 60 * 12,  # wait up to 12 hours
        mode="reschedule",
    )

    convert_jsons = PythonOperator(
        task_id="convert_jsons_to_csv",
        python_callable=jsons_to_csv,
    )

    # retrain_model = DockerOperator(
    #     task_id="retrain_model",
    #     image=TRAINING_IMAGE,
    #     docker_url="unix://var/run/docker.sock",
    #     mounts=[
    #         Mount(
    #             source="/data",
    #             target="/data",
    #             type="bind",
    #             read_only=False,
    #         )
    #     ],
    #     mount_tmp_dir=False,
    #     environment=get_mlflow_env(),
    #     auto_remove=True,
    # )

    wait_for_samples >> convert_jsons  # >> retrain_model
