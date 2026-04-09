from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.serving.mlflow_bento import (
    DEFAULT_MLFLOW_MODEL_NAME,
    DEFAULT_TRAINING_MANIFEST_PATH,
    find_model_version_by_run_id,
    make_mlflow_client,
    model_version_uri,
    write_json_file,
)
from src.serving.prepare_bento_assets import ensure_local_text_backbone
from src.training.mlflow_text_pyfunc import TextClassifierPyFuncModel


def register_text_model_in_mlflow(
    *,
    model_weights_path: str | Path,
    train_config_path: str | Path,
    preprocessing_config_path: str | Path,
    label_encoding_path: str | Path,
    registered_model_name: str = DEFAULT_MLFLOW_MODEL_NAME,
    registration_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST_PATH,
    run_id: str | None = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    backbone_dir: str | Path | None = None,
    mlflow_module=None,
    client=None,
) -> dict[str, Any]:
    if mlflow_module is None:  # pragma: no cover - exercised when mlflow is installed
        import mlflow as mlflow_module

    active_run = mlflow_module.active_run()
    resolved_run_id = run_id or (active_run.info.run_id if active_run is not None else None)
    if not resolved_run_id:
        raise RuntimeError("register_text_model_in_mlflow requires an active MLflow run or an explicit run_id.")

    client = client or make_mlflow_client(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        mlflow_module=mlflow_module,
    )

    train_config_path = Path(train_config_path)
    preprocessing_config_path = Path(preprocessing_config_path)
    backbone_dir = Path(backbone_dir) if backbone_dir is not None else ensure_local_text_backbone(
        train_config_path=train_config_path,
        preprocessing_config_path=preprocessing_config_path,
    )

    artifacts = {
        "weights": str(Path(model_weights_path).resolve()),
        "train_config": str(train_config_path.resolve()),
        "preprocessing_config": str(preprocessing_config_path.resolve()),
        "label_encoding": str(Path(label_encoding_path).resolve()),
        "backbone": str(Path(backbone_dir).resolve()),
    }

    input_example = pd.DataFrame([
        {
            "designation": "robe femme",
            "description": "bleu",
            "top_k": 3,
        }
    ])

    model_info = mlflow_module.pyfunc.log_model(
        artifact_path="text_classifier_model",
        python_model=TextClassifierPyFuncModel(),
        artifacts=artifacts,
        code_paths=[str(Path(__file__).resolve().parents[2])],
        input_example=input_example,
        registered_model_name=registered_model_name,
        metadata={
            "task": "text-classification",
            "training_entrypoint": "src.training.run_text_training",
        },
    )

    registered_model_version = getattr(model_info, "registered_model_version", None)
    if registered_model_version is None:
        registered_model_version = find_model_version_by_run_id(
            client=client,
            model_name=registered_model_name,
            run_id=resolved_run_id,
        )

    if registered_model_version is not None:
        client.set_model_version_tag(
            name=registered_model_name,
            version=str(registered_model_version),
            key="validation_status",
            value="pending",
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=str(registered_model_version),
            key="source_run_id",
            value=resolved_run_id,
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=str(registered_model_version),
            key="serving_flavor",
            value="mlflow.pyfunc",
        )

    manifest = {
        "mlflow_model_name": registered_model_name,
        "mlflow_model_uri": model_version_uri(registered_model_name, registered_model_version)
        if registered_model_version is not None
        else None,
        "mlflow_run_id": resolved_run_id,
        "mlflow_version": str(registered_model_version) if registered_model_version is not None else None,
        "model_artifact_path": getattr(model_info, "model_uri", None),
        "validation_status": "pending",
    }
    write_json_file(registration_manifest_path, manifest)

    if hasattr(mlflow_module, "log_dict"):
        mlflow_module.log_dict(manifest, "manifests/mlflow_text_model.json")

    return manifest
