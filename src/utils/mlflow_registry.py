from datetime import datetime, timezone
import mlflow
from mlflow.tracking import MlflowClient


def get_production_model_timestamp(model_name: str) -> datetime:
    client = MlflowClient()

    versions = client.get_latest_versions(
        name=model_name,
        stages=["Production"]
    )
    if not versions:
        raise RuntimeError(f"No Production model found for {model_name}")

    prod_version = versions[0]

    # MLflow stores timestamps in milliseconds since epoch
    if prod_version.last_updated_timestamp is None:
        raise RuntimeError("Production model has no last_updated_timestamp")

    return datetime.fromtimestamp(
        prod_version.last_updated_timestamp / 1000,
        tz=timezone.utc,
    )
