from __future__ import annotations

import json

from src.serving.mlflow_bento import DEFAULT_MLFLOW_ALIAS, DEFAULT_MLFLOW_MODEL_NAME
from src.serving.sync_mlflow_to_bento import sync_mlflow_model_to_bento


def register_text_model(
    model_name: str = DEFAULT_MLFLOW_MODEL_NAME,
    alias: str = DEFAULT_MLFLOW_ALIAS,
    manifest_path: str = "artifacts/deployment_manifest.json",
) -> dict:
    """Wrapper around new MLflow -> BentoML sync flow:"""
    return sync_mlflow_model_to_bento(
        model_name=model_name,
        alias=alias,
        manifest_path=manifest_path,
    )


if __name__ == "__main__":
    manifest = register_text_model()
    print(json.dumps(manifest, indent=2, sort_keys=True))
