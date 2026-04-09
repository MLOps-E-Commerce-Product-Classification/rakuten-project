from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MLFLOW_MODEL_NAME = "text-classifier"
DEFAULT_MLFLOW_ALIAS = "production"
DEFAULT_PROMOTION_METRIC = "eval_macro_f1"
DEFAULT_DEPLOYMENT_MANIFEST_PATH = Path("artifacts/deployment_manifest.json")
DEFAULT_TRAINING_MANIFEST_PATH = Path("artifacts/mlflow_text_model.json")
DEFAULT_BENTO_FALLBACK_TAG = f"{DEFAULT_MLFLOW_MODEL_NAME}:latest"


def _import_mlflow():
    try:
        import mlflow
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised in integration environments
        raise RuntimeError(
            "mlflow is required for the MLflow/BentoML registry workflow. "
            "Install the repository with the 'data' extra or use 'uv sync --all-extras'."
        ) from exc
    return mlflow


def _import_bentoml():
    try:
        import bentoml
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised in integration environments
        raise RuntimeError(
            "bentoml is required for the MLflow/BentoML serving workflow. "
            "Install the repository with the 'api' extra or use 'uv sync --all-extras'."
        ) from exc
    return bentoml


def read_json_file(
    path: str | Path, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {} if default is None else dict(default)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path: str | Path, payload: dict[str, Any]) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return file_path


def configure_mlflow(
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    *,
    mlflow_module=None,
):
    mlflow_module = mlflow_module or _import_mlflow()
    effective_tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    effective_registry_uri = registry_uri or os.environ.get("MLFLOW_REGISTRY_URI")

    if effective_tracking_uri:
        mlflow_module.set_tracking_uri(effective_tracking_uri)
    if effective_registry_uri:
        mlflow_module.set_registry_uri(effective_registry_uri)

    return mlflow_module


def make_mlflow_client(
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    *,
    mlflow_module=None,
):
    mlflow_module = configure_mlflow(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        mlflow_module=mlflow_module,
    )
    return mlflow_module.MlflowClient()


def model_alias_uri(model_name: str, alias: str) -> str:
    return f"models:/{model_name}@{alias}"


def model_version_uri(model_name: str, version: str | int) -> str:
    return f"models:/{model_name}/{version}"


def load_run_metrics(client, run_id: str | None) -> dict[str, float]:
    if not run_id:
        return {}
    run = client.get_run(run_id)
    data = getattr(run, "data", None)
    metrics = getattr(data, "metrics", {}) if data is not None else {}
    return {str(key): float(value) for key, value in dict(metrics).items()}


def build_metric_candidates(primary_metric: str) -> list[str]:
    candidates: list[str] = []
    for item in [
        primary_metric,
        "eval_macro_f1",
        "final_best_val_macro_f1",
        "best_macro_f1",
        "val_macro_f1",
    ]:
        if item and item not in candidates:
            candidates.append(item)
    return candidates


def resolve_metric_value(
    metrics: dict[str, float], metric_names: Iterable[str]
) -> tuple[str | None, float | None]:
    for name in metric_names:
        if name in metrics:
            return name, float(metrics[name])
    return None, None


def extract_model_version_tags(model_version) -> dict[str, str]:
    tags = getattr(model_version, "tags", None)
    if tags is None and hasattr(model_version, "__dict__"):
        tags = model_version.__dict__.get("tags")
    return {str(key): str(value) for key, value in dict(tags or {}).items()}


def find_model_version_by_run_id(client, model_name: str, run_id: str) -> str | None:
    for model_version in client.search_model_versions(f"name='{model_name}'"):
        if str(getattr(model_version, "run_id", "")) == str(run_id):
            return str(getattr(model_version, "version"))
    return None


def resolve_candidate_version(
    client, model_name: str, current_champion_version: str | None = None
) -> Any:
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    if not versions:
        raise ValueError(f"No registered versions found for model '{model_name}'.")

    versions.sort(key=lambda item: int(str(getattr(item, "version"))))
    for model_version in reversed(versions):
        version = str(getattr(model_version, "version"))
        if current_champion_version is None or version != str(current_champion_version):
            return model_version
    raise ValueError(
        f"No candidate version available for model '{model_name}' besides the current champion."
    )


def get_model_version_by_alias(client, model_name: str, alias: str):
    return client.get_model_version_by_alias(name=model_name, alias=alias)


def find_existing_bento_model(
    bentoml_module, bento_model_name: str, mlflow_model_name: str, mlflow_version: str
):
    """
    Return a bentoml model object if a model with name `bento_model_name` and matching
    mlflow metadata (mlflow_model_name + mlflow_version) exists in the local BentoML store.
    If no such model exists, return None. This function safely handles BentoML's NotFound.
    """
    try:
        try:
            models = bentoml_module.models.list(bento_model_name)
        except TypeError:
            models = bentoml_module.models.list()
    except Exception as e:
        try:
            from bentoml.exceptions import NotFound as BentoNotFound

            if isinstance(e, BentoNotFound):
                return None
        except Exception:
            pass
        raise

    for model in models:
        tag = str(getattr(model, "tag", ""))
        if not tag.startswith(f"{bento_model_name}:"):
            continue

        metadata = dict(getattr(model, "metadata", {}) or {})
        if str(metadata.get("mlflow_model_name")) == str(mlflow_model_name) and str(
            metadata.get("mlflow_version")
        ) == str(mlflow_version):
            return model
    return None


def resolve_bento_model_reference(
    manifest_path: str | Path = DEFAULT_DEPLOYMENT_MANIFEST_PATH,
    *,
    env: dict[str, str] | None = None,
    fallback_tag: str = DEFAULT_BENTO_FALLBACK_TAG,
) -> dict[str, Any]:
    env = env or os.environ
    manifest_file = Path(env.get("BENTO_DEPLOYMENT_MANIFEST", str(manifest_path)))
    manifest = read_json_file(manifest_file)
    model_tag = (
        env.get("BENTO_MODEL_TAG") or manifest.get("bentoml_model_tag") or fallback_tag
    )

    return {
        "model_tag": model_tag,
        "manifest_path": str(manifest_file),
        "manifest_present": manifest_file.exists(),
        "mlflow_model_name": manifest.get("mlflow_model_name"),
        "mlflow_alias": manifest.get("mlflow_alias"),
        "mlflow_version": manifest.get("mlflow_version"),
        "mlflow_run_id": manifest.get("mlflow_run_id"),
        "mlflow_model_uri": manifest.get("mlflow_model_uri"),
        "validation_status": manifest.get("validation_status"),
    }
