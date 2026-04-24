import requests
import json
import os
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timezone

NEW_DATA_DIR = Path("/app/data/new_data")
PROCESSED_DIR = Path("/app/data/new_train_data")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://bentoml:3000"
LOGIN_URL = f"{BASE_URL}/login"
PREDICT_URL = f"{BASE_URL}/predict"


def process_new_data():
    files = list(NEW_DATA_DIR.glob("*.json"))
    if not files:
        return

    # Authentication
    payload = {"credentials": {"username": "user123", "password": "password123"}}

    try:
        login_res = requests.post(LOGIN_URL, json=payload, timeout=5)
        login_res.raise_for_status()
        token = login_res.json()["token"]
    except Exception as e:
        raise RuntimeError(f"Authentication failed: {e}")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Processing loop
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)

        try:
            predict_payload = {"input_data": data}
            response = requests.post(
                PREDICT_URL, json=predict_payload, headers=headers, timeout=15
            )
            response.raise_for_status()
            prediction_result = response.json()

            # Extract label and ensure integer type to avoid JSON float issues
            raw_label = prediction_result.get("predicted_rakuten_code")
            label = int(raw_label) if raw_label is not None else None

            # Prepare output object
            combined = {
                "designation": data.get("designation"),
                "description": data.get("description"),
                "label": label,
                "source": "pseudo-label",
                "timestamp": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }

            out_file = PROCESSED_DIR / file.name
            with open(out_file, "w") as out_f:
                json.dump(combined, out_f, indent=2)

            os.remove(file)
            print(f"Successfully processed: {file.name}")

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            continue


with DAG(
    dag_id="infere_new_data",
    tags=["rakuten"],
    schedule_interval="* * * * *",
    start_date=datetime(2026, 3, 29),
    catchup=False,
) as dag:
    wait_for_files = FileSensor(
        task_id="wait_for_new_json",
        filepath=str(NEW_DATA_DIR),
        mode="reschedule",
        poke_interval=30,
        timeout=120,
    )

    process_files = PythonOperator(
        task_id="process_files", python_callable=process_new_data
    )

    wait_for_files >> process_files
