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

API_URL = "http://localhost:8000/predict"


def process_new_data():
    files = list(NEW_DATA_DIR.glob("*.json"))

    for file in files:
        # load JSON
        with open(file, "r") as f:
            data = json.load(f)

        # send to API
        try:
            response = requests.post(API_URL, json=data, timeout=10)
            response.raise_for_status()
            prediction = response.json()
        except Exception as e:
            raise RuntimeError(f"API call failed for {file}: {e}")

        # save combined JSON + prediction
        out_file = PROCESSED_DIR / file.name
        combined = {"designation": data["designation"], 
                    "description": data["description"],
                    "label": prediction, #TODO fetch the correct part of the prediction
                    "source": "pseudo-label",
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z") #TODO ensure models are saved with correct timestamp
                    }

        with open(out_file, "w") as f:
            json.dump(combined, f, indent=2)

        # delete original
        os.remove(file)


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
        poke_interval=30,  # check every 30 seconds
        timeout=120,  # stop waiting after 1 min if no file
    )

    process_files = PythonOperator(
        task_id="process_files", python_callable=process_new_data
    )

    wait_for_files >> process_files
