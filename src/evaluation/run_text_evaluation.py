from __future__ import annotations

import argparse
import os
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
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
        data = json.load(f)
    if "code_to_idx" in data:
        return data["code_to_idx"]
    return data


def load_full_label_encoding(mapping_path: str | Path) -> dict:
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_dataframe_columns(df, required_columns, df_name):
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {sorted(missing_columns)}"
        )


def apply_label_mapping(df, label_col, mapping, encoded_label_col="label"):
    df = df.copy()
    labels_as_str = df[label_col].astype(str)
    unseen_labels = sorted(set(labels_as_str) - set(mapping.keys()))
    if unseen_labels:
        raise ValueError(
            "Evaluation set contains labels not present in the training label mapping: "
            f"{unseen_labels[:10]}" + (" ..." if len(unseen_labels) > 10 else "")
        )
    df[encoded_label_col] = labels_as_str.map(mapping)
    df[encoded_label_col] = df[encoded_label_col].astype(int)
    return df


def generate_evaluation_plots(
    results: dict,
    output_dir: Path,
    idx_to_name: dict[str, str] | None = None,
) -> None:
    """
    Generate and save evaluation plots.
    If idx_to_name is provided, class labels are shown as human-readable names.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = results.get("confusion_matrix")
    per_class_f1 = results.get("per_class_f1", {})

    def _label(idx) -> str:
        if idx_to_name:
            return idx_to_name.get(str(idx), str(idx))
        return str(idx)

    # --- 1. Confusion Matrix (row-normalized) ---
    if cm:
        cm_array = np.array(cm)
        cm_norm = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True).clip(
            min=1
        )
        n = cm_norm.shape[0]
        fig, ax = plt.subplots(figsize=(max(10, n * 0.5), max(8, n * 0.45)))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Confusion Matrix (row-normalized)", fontsize=14, pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ticks = list(range(n))
        tick_labels = [_label(i) for i in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
        plt.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[Plots] Saved confusion_matrix.png")

    # --- 2. Per-Class F1 Bar Chart ---
    if per_class_f1:
        classes = [_label(k) for k in per_class_f1.keys()]
        scores = list(per_class_f1.values())
        fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.5), 5))
        colors = [
            "#d9534f" if s < 0.6 else "#f0ad4e" if s < 0.8 else "#5cb85c"
            for s in scores
        ]
        ax.bar(classes, scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(
            y=float(np.mean(scores)),
            color="steelblue",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean F1: {np.mean(scores):.3f}",
        )
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-Class F1 Score", fontsize=14)
        ax.set_xlabel("Class", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=8)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(output_dir / "per_class_f1.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[Plots] Saved per_class_f1.png")

    # --- 3. Classification Report Table ---
    if per_class_f1:
        macro_f1 = results.get("macro_f1", float(np.mean(list(per_class_f1.values()))))
        accuracy = results.get("accuracy", None)
        rows = [[_label(cls), f"{score:.4f}"] for cls, score in per_class_f1.items()]
        rows.append(["— macro avg —", f"{macro_f1:.4f}"])
        if accuracy is not None:
            rows.append(["— accuracy —", f"{accuracy:.4f}"])
        col_labels = ["Class", "F1 Score"]
        fig, ax = plt.subplots(figsize=(4, max(4, len(rows) * 0.28 + 1)))
        ax.axis("off")
        table = ax.table(
            cellText=rows, colLabels=col_labels, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.1)
        ax.set_title("Classification Report", fontsize=13, pad=10)
        plt.tight_layout()
        fig.savefig(
            output_dir / "classification_report.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print("[Plots] Saved classification_report.png")


def log_evaluation_to_mlflow(results, results_output_path, mlflow_run_id):
    plots_dir = results_output_path.parent
    with mlflow.start_run(run_id=mlflow_run_id):
        metrics = results.get("metrics", {})
        for key in ["macro_f1", "weighted_f1", "accuracy", "loss"]:
            if key in metrics:
                mlflow.log_metric(f"eval_{key}", float(metrics[key]))
        for cls, score in metrics.get("per_class_f1", {}).items():
            mlflow.log_metric(f"eval_f1_class_{cls}", float(score))
        for plot_file in [
            "confusion_matrix.png",
            "per_class_f1.png",
            "classification_report.png",
        ]:
            plot_path = plots_dir / plot_file
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), artifact_path="evaluation_plots")
            else:
                print(f"[MLflow] Plot not found, skipping: {plot_path}")
        if results_output_path.exists():
            mlflow.log_artifact(
                str(results_output_path), artifact_path="evaluation_plots"
            )
    print(f"[MLflow] Evaluation results logged to run: {mlflow_run_id}")


def run_text_evaluation(
    x_data_csv_path: str | Path,
    y_data_csv_path: str | Path,
    split_ids_dir: str | Path | None = None,
    train_config_path: str | Path = "configs/text_train_config.yaml",
    eval_config_path: str | Path = "configs/text_evaluate_config.yaml",
    preprocessing_config_path: str | Path = "configs/text_preprocessing_config.yaml",
    model_weights_path: str | Path = "models/best_text_model.pt",
    label_encoding_path: str | Path = "artifacts/label_mapping.json",
    results_output_path: str | Path = "results/text_evaluation_results.json",
    mlflow_run_id: str | None = None,
) -> dict:
    # 1. Prepare paths
    x_data_csv_path = Path(x_data_csv_path)
    y_data_csv_path = Path(y_data_csv_path)
    model_weights_path = Path(model_weights_path)
    label_encoding_path = Path(label_encoding_path)
    results_output_path = Path(results_output_path)

    # 2. Validate inputs
    if not x_data_csv_path.exists():
        raise FileNotFoundError(f"X data CSV not found: {x_data_csv_path}")
    if not y_data_csv_path.exists():
        raise FileNotFoundError(f"Y data CSV not found: {y_data_csv_path}")
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    # 3. Load configs and label mapping
    train_config = load_config(train_config_path)
    label_mapping = load_label_mapping(label_encoding_path)

    # 3b. Load idx_to_name for human-readable plot labels
    full_encoding = load_full_label_encoding(label_encoding_path)
    idx_to_name = {str(k): v for k, v in full_encoding.get("idx_to_name", {}).items()}

    training_config = train_config.get("training", {})
    model_config = train_config.get("model", {})
    data_config = train_config.get("data", {})

    # 4. Load data
    if x_data_csv_path == y_data_csv_path:
        eval_df = pd.read_csv(x_data_csv_path)
    else:
        x_df = pd.read_csv(x_data_csv_path)
        y_df = pd.read_csv(y_data_csv_path)
        eval_df = pd.concat([x_df, y_df], axis=1)

    # 5. Column configuration
    designation_col = data_config.get("designation_col", "designation")
    description_col = data_config.get("description_col", "description")
    label_col = data_config.get("label_col", "prdtypecode")
    encoded_label_col = data_config.get("encoded_label_col", "label")
    return_quality_report = bool(data_config.get("return_quality_report", False))

    # 6. Validate columns
    required_columns = {designation_col}
    if encoded_label_col not in eval_df.columns:
        required_columns.add(label_col)
    validate_dataframe_columns(eval_df, required_columns, "Merged Evaluation Data")
    if description_col not in eval_df.columns:
        eval_df[description_col] = ""

    # 7. Apply label mapping
    if encoded_label_col in eval_df.columns and pd.api.types.is_integer_dtype(
        eval_df[encoded_label_col]
    ):
        print(
            f"Column '{encoded_label_col}' already exists and is numeric. Using existing labels."
        )
    else:
        eval_df = apply_label_mapping(
            eval_df,
            label_col=label_col,
            mapping=label_mapping,
            encoded_label_col=encoded_label_col,
        )

    # 8. Build dataset and dataloader
    model_name = model_config.get("name", "bert-base-multilingual-cased")
    num_classes = len(label_mapping)
    identity_encoding = {"code_to_idx": {str(i): i for i in range(num_classes)}}

    eval_dataset = RakutenTextDataset(
        dataframe=eval_df,
        config_path=preprocessing_config_path,
        label_encoding=identity_encoding,
        designation_col=designation_col,
        description_col=description_col,
        label_col=encoded_label_col,
        return_quality_report=return_quality_report,
    )

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 9. Build model and load weights
    model = build_text_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 10. Run evaluation
    criterion = torch.nn.CrossEntropyLoss()
    results = evaluate_model(
        model=model,
        dataloader=eval_dataloader,
        criterion=criterion,
        device=device,
        config_path=eval_config_path,
        num_classes=num_classes,
    )

    # 11. Save metadata
    results["metadata"] = {
        "model_name": model_name,
        "num_classes": num_classes,
        "x_data_csv_path": str(x_data_csv_path),
        "y_data_csv_path": str(y_data_csv_path),
        "split_ids_dir": str(split_ids_dir) if split_ids_dir else None,
        "model_weights_path": str(model_weights_path),
    }
    save_evaluation_results(results, results_output_path)

    # 12. Generate evaluation plots (with label names)
    plots_dir = results_output_path.parent
    generate_evaluation_plots(results["metrics"], plots_dir, idx_to_name=idx_to_name)

    # 13. Log to MLflow (optional)
    if mlflow_run_id:
        log_evaluation_to_mlflow(
            results=results,
            results_output_path=results_output_path,
            mlflow_run_id=mlflow_run_id,
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text model evaluation.")
    parser.add_argument("--x_data_csv_path", default="data/raw/X_val.csv")
    parser.add_argument("--y_data_csv_path", default="data/raw/Y_val.csv")
    parser.add_argument("--split_ids_dir", default="artifacts/splits")
    parser.add_argument("--train_config_path", default="configs/text_train_config.yaml")
    parser.add_argument(
        "--eval_config_path", default="configs/text_evaluate_config.yaml"
    )
    parser.add_argument(
        "--preprocessing_config_path", default="configs/text_preprocessing_config.yaml"
    )
    parser.add_argument("--model_weights_path", default="models/best_text_model.pt")
    parser.add_argument("--label_encoding_path", default="artifacts/label_mapping.json")
    parser.add_argument(
        "--results_output_path", default="results/text_evaluation_results.json"
    )
    parser.add_argument("--mlflow_run_id", default=None)
    args = parser.parse_args()

    mlflow_run_id = args.mlflow_run_id or os.environ.get("MLFLOW_RUN_ID")

    results = run_text_evaluation(
        x_data_csv_path=args.x_data_csv_path,
        y_data_csv_path=args.y_data_csv_path,
        split_ids_dir=args.split_ids_dir,
        train_config_path=args.train_config_path,
        eval_config_path=args.eval_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        model_weights_path=args.model_weights_path,
        label_encoding_path=args.label_encoding_path,
        results_output_path=args.results_output_path,
        mlflow_run_id=mlflow_run_id,
    )

    print("Evaluation finished.")
    print(f"Main metric: {results['main_metric']} = {results['main_metric_value']:.4f}")
    print("Results saved to: results/text_evaluation_results.json")
