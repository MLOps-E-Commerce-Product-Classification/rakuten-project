import json
import pandas as pd
import random
from pathlib import Path 
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

NEW_DATA_DIR = Path("data/new_data")


def sample_data():
    df = pd.read_csv("/app/data/raw/X_test_update.csv")

    n = random.randint(1, 10)
    sample = df.sample(n)

    output_dir = Path(NEW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, r in sample.iterrows():

        row = {
            "designation": r["designation"],
            "description": r["description"]
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = output_dir / f"sample_{timestamp}_{idx}.json"

        with open(filename, "w") as f:
            json.dump(row, f)

with DAG(
    dag_id="simulate_new_data",
    tags=['rakuten'],
    schedule_interval="* * * * *",
    start_date=datetime(2026,3,29),
    catchup=False
) as dag:

    sim_new_data = PythonOperator(
        task_id="sim_new_data",
        python_callable=sample_data
    )

     