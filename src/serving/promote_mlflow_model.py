from __future__ import annotations

import argparse
import json

from src.serving.mlflow_bento import (
    DEFAULT_MLFLOW_ALIAS,
    DEFAULT_MLFLOW_MODEL_NAME,
    DEFAULT_PROMOTION_METRIC,
    build_metric_candidates,
    extract_model_version_tags,
    get_model_version_by_alias,
    load_run_metrics,
    make_mlflow_client,
    resolve_candidate_version,
    resolve_metric_value,
    write_json_file,
)


def promote_model(
    *,
    model_name: str = DEFAULT_MLFLOW_MODEL_NAME,
    alias: str = DEFAULT_MLFLOW_ALIAS,
    candidate_version: str | None = None,
    metric_name: str = DEFAULT_PROMOTION_METRIC,
    min_improvement: float = 0.0,
    required_tag: str | None = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    output_path: str | None = None,
    client=None,
    mlflow_module=None,
) -> dict:
    client = client or make_mlflow_client(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        mlflow_module=mlflow_module,
    )

    champion_version = None
    champion_metric_value = None
    champion_metric_name = None
    try:
        champion_version = get_model_version_by_alias(client, model_name, alias)
        champion_metrics = load_run_metrics(client, getattr(champion_version, "run_id", None))
        champion_metric_name, champion_metric_value = resolve_metric_value(
            champion_metrics,
            build_metric_candidates(metric_name),
        )
    except Exception:
        champion_version = None

    if candidate_version is None:
        candidate = resolve_candidate_version(
            client=client,
            model_name=model_name,
            current_champion_version=getattr(champion_version, "version", None),
        )
    else:
        candidate = client.get_model_version(name=model_name, version=str(candidate_version))

    candidate_tags = extract_model_version_tags(candidate)
    if required_tag is not None:
        key, expected_value = required_tag.split("=", maxsplit=1)
        actual_value = candidate_tags.get(key)
        if actual_value != expected_value:
            result = {
                "promoted": False,
                "reason": f"Candidate version {candidate.version} is missing required tag {required_tag}.",
                "mlflow_model_name": model_name,
                "mlflow_alias": alias,
                "candidate_version": str(candidate.version),
                "champion_version": str(getattr(champion_version, 'version', '')) or None,
            }
            if output_path:
                write_json_file(output_path, result)
            return result

    candidate_metrics = load_run_metrics(client, getattr(candidate, "run_id", None))
    candidate_metric_name, candidate_metric_value = resolve_metric_value(
        candidate_metrics,
        build_metric_candidates(metric_name),
    )
    if candidate_metric_value is None:
        raise ValueError(
            f"Candidate version {candidate.version} does not expose any of the expected metrics: "
            f"{build_metric_candidates(metric_name)}"
        )

    should_promote = champion_metric_value is None or candidate_metric_value >= champion_metric_value + float(min_improvement)
    if should_promote:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(candidate.version),
        )

    result = {
        "promoted": bool(should_promote),
        "mlflow_model_name": model_name,
        "mlflow_alias": alias,
        "candidate_version": str(candidate.version),
        "candidate_run_id": getattr(candidate, "run_id", None),
        "candidate_metric_name": candidate_metric_name,
        "candidate_metric_value": candidate_metric_value,
        "champion_version": str(getattr(champion_version, "version", "")) or None,
        "champion_run_id": getattr(champion_version, "run_id", None),
        "champion_metric_name": champion_metric_name,
        "champion_metric_value": champion_metric_value,
        "min_improvement": float(min_improvement),
        "validation_status": candidate_tags.get("validation_status"),
    }

    if output_path:
        write_json_file(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote a registered MLflow model version to the chosen alias.")
    parser.add_argument("--model-name", default=DEFAULT_MLFLOW_MODEL_NAME)
    parser.add_argument("--alias", default=DEFAULT_MLFLOW_ALIAS)
    parser.add_argument("--candidate-version", default=None)
    parser.add_argument("--metric-name", default=DEFAULT_PROMOTION_METRIC)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--required-tag", default=None, help="Optional gate like validation_status=approved")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--registry-uri", default=None)
    parser.add_argument("--output-path", default="artifacts/mlflow_promotion_manifest.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = promote_model(
        model_name=args.model_name,
        alias=args.alias,
        candidate_version=args.candidate_version,
        metric_name=args.metric_name,
        min_improvement=args.min_improvement,
        required_tag=args.required_tag,
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        output_path=args.output_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
