from __future__ import annotations

import argparse
import json

from src.serving.mlflow_bento import (
    DEFAULT_DEPLOYMENT_MANIFEST_PATH,
    DEFAULT_MLFLOW_ALIAS,
    DEFAULT_MLFLOW_MODEL_NAME,
    extract_model_version_tags,
    find_existing_bento_model,
    get_model_version_by_alias,
    make_mlflow_client,
    model_alias_uri,
    write_json_file,
    _import_bentoml,
)


def sync_mlflow_model_to_bento(
    *,
    model_name: str = DEFAULT_MLFLOW_MODEL_NAME,
    alias: str = DEFAULT_MLFLOW_ALIAS,
    bento_model_name: str | None = None,
    manifest_path: str = str(DEFAULT_DEPLOYMENT_MANIFEST_PATH),
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    client=None,
    bentoml_module=None,
    mlflow_module=None,
) -> dict:
    client = client or make_mlflow_client(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        mlflow_module=mlflow_module,
    )
    bentoml_module = bentoml_module or _import_bentoml()
    bento_model_name = bento_model_name or model_name

    model_version = get_model_version_by_alias(client, model_name, alias)
    version_tags = extract_model_version_tags(model_version)
    resolved_version = str(model_version.version)
    source_uri = model_alias_uri(model_name, alias)

    existing_model = find_existing_bento_model(
        bentoml_module=bentoml_module,
        bento_model_name=bento_model_name,
        mlflow_model_name=model_name,
        mlflow_version=resolved_version,
    )

    updated = False
    bento_model = existing_model
    if bento_model is None:
        labels = {
            "registry": "mlflow",
            "mlflow_model_name": model_name,
            "mlflow_alias": alias,
        }

        raw_metadata = {
            "mlflow_model_name": model_name,
            "mlflow_alias": alias,
            "mlflow_version": resolved_version,
            "mlflow_run_id": getattr(model_version, "run_id", None),
            "mlflow_model_uri": source_uri,
            "validation_status": version_tags.get("validation_status"),
        }

        metadata = {}
        for k, v in raw_metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, complex, bool)):
                metadata[k] = v
            else:
                metadata[k] = str(v)

        try:
            bento_model = bentoml_module.mlflow.import_model(
                bento_model_name,
                source_uri,
                labels=labels,
                metadata=metadata,
            )
            updated = True
        except Exception as e:
            print("Fehler beim BentoML-Import:", type(e), e)
            print("Import-URI:", source_uri)
            print("Labels:", labels)
            print("Metadata (clean):", metadata)
            raise

    manifest = {
        "updated": updated,
        "mlflow_model_name": model_name,
        "mlflow_alias": alias,
        "mlflow_version": resolved_version,
        "mlflow_run_id": getattr(model_version, "run_id", None),
        "mlflow_model_uri": source_uri,
        "bentoml_model_tag": str(bento_model.tag),
        "validation_status": version_tags.get("validation_status"),
    }
    write_json_file(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize the champion MLflow model into BentoML."
    )
    parser.add_argument("--model-name", default=DEFAULT_MLFLOW_MODEL_NAME)
    parser.add_argument("--alias", default=DEFAULT_MLFLOW_ALIAS)
    parser.add_argument("--bento-model-name", default=None)
    parser.add_argument(
        "--manifest-path", default=str(DEFAULT_DEPLOYMENT_MANIFEST_PATH)
    )
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--registry-uri", default=None)
    return parser


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    import mlflow
    import os


    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    args = build_parser().parse_args()
    manifest = sync_mlflow_model_to_bento(
        model_name=args.model_name,
        alias=args.alias,
        bento_model_name=args.bento_model_name,
        manifest_path=args.manifest_path,
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
