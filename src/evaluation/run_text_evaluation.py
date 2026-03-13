from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.text_dataset import RakutenTextDataset
from src.evaluation.text_evaluate import evaluate_model, save_evaluation_results
from src.models.text_classifier import build_text_model


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_mapping(mapping_path: str | Path) -> dict[str, int]:
    mapping_path = Path(mapping_path)

    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    df_name: str,
) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {sorted(missing_columns)}"
        )


def apply_label_mapping(
    df: pd.DataFrame,
    label_col: str,
    mapping: dict[str, int],
    encoded_label_col: str = "label",
) -> pd.DataFrame:
    df = df.copy()

    labels_as_str = df[label_col].astype(str)
    unseen_labels = sorted(set(labels_as_str) - set(mapping.keys()))
    if unseen_labels:
        raise ValueError(
            "Evaluation set contains labels not present in the training label mapping: "
            f"{unseen_labels[:10]}"
            + (" ..." if len(unseen_labels) > 10 else "")
        )

    df[encoded_label_col] = labels_as_str.map(mapping)
    df[encoded_label_col] = df[encoded_label_col].astype(int)

    return df


def run_text_evaluation(
    eval_csv_path: str | Path,
    train_config_path: str | Path = "configs/text_train_config.yaml",
    eval_config_path: str | Path = "configs/text_evaluate_config.yaml",
    preprocessing_config_path: str | Path = "configs/text_preprocessing_config.yaml",
    model_weights_path: str | Path = "models/best_text_model.pt",
    label_mapping_path: str | Path = "artifacts/label_mapping.json",
    results_output_path: str | Path = "results/text_evaluation_results.json",
) -> dict:
    """
    Run end-to-end text evaluation.
    """
    train_config = load_config(train_config_path)
    eval_config = load_config(eval_config_path)
    label_mapping = load_label_mapping(label_mapping_path)

    training_config = train_config.get("training", {})
    model_config = train_config.get("model", {})
    data_config = train_config.get("data", {})
    metrics_config = eval_config.get("metrics", {})

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))

    model_name = model_config.get("name", "bert-base-multilingual-cased")

    designation_col = data_config.get("designation_col", "designation")
    description_col = data_config.get("description_col", "description")
    label_col = data_config.get("label_col", "prdtypecode")
    encoded_label_col = data_config.get("encoded_label_col", "label")
    return_quality_report = bool(data_config.get("return_quality_report", False))

    compute_per_class_f1 = bool(metrics_config.get("compute_per_class_f1", True))
    main_metric = metrics_config.get("main_metric", "macro_f1")

    eval_csv_path = Path(eval_csv_path)
    model_weights_path = Path(model_weights_path)
    results_output_path = Path(results_output_path)

    if not eval_csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {eval_csv_path}")
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    eval_df = pd.read_csv(eval_csv_path)

    required_columns = {designation_col, label_col}
    validate_dataframe_columns(eval_df, required_columns, "Evaluation CSV")

    if description_col not in eval_df.columns:
        eval_df[description_col] = ""

    eval_df = apply_label_mapping(
        eval_df,
        label_col=label_col,
        mapping=label_mapping,
        encoded_label_col=encoded_label_col,
    )

    eval_dataset = RakutenTextDataset(
        dataframe=eval_df,
        config_path=preprocessing_config_path,
        designation_col=designation_col,
        description_col=description_col,
        label_col=encoded_label_col,
        return_quality_report=return_quality_report,
    )

    pin_memory = torch.cuda.is_available()

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    num_classes = len(label_mapping)

    model = build_text_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss()

    results = evaluate_model(
        model=model,
        dataloader=eval_dataloader,
        criterion=criterion,
        device=device,
        config_path=eval_config_path,
        num_classes=num_classes,
    )

    results["metadata"] = {
        "model_name": model_name,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "compute_per_class_f1": compute_per_class_f1,
        "main_metric": main_metric,
        "model_weights_path": str(model_weights_path),
        "label_mapping_path": str(label_mapping_path),
        "eval_csv_path": str(eval_csv_path),
    }

    save_evaluation_results(results, results_output_path)

    return results


if __name__ == "__main__":
    results = run_text_evaluation(
        eval_csv_path="data/val_split.csv",
        train_config_path="configs/text_train_config.yaml",
        eval_config_path="configs/text_evaluate_config.yaml",
        preprocessing_config_path="configs/text_preprocessing_config.yaml",
        model_weights_path="models/best_text_model.pt",
        label_mapping_path="artifacts/label_mapping.json",
        results_output_path="results/text_evaluation_results.json",
    )

    print("Evaluation finished.")
    print(f"Main metric: {results['main_metric']} = {results['main_metric_value']:.4f}")
    print("Results saved to: results/text_evaluation_results.json")
