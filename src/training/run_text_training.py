from pathlib import Path
import json
import os
import subprocess
from datetime import datetime

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data.text_dataset import RakutenTextDataset
from src.models.text_classifier import build_text_model
from src.training.train_text import train_model

load_dotenv()


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_encoding(label_encoding_path: str | Path) -> dict:
    label_encoding_path = Path(label_encoding_path)
    if not label_encoding_path.exists():
        raise FileNotFoundError(f"Label encoding file not found: {label_encoding_path}")
    with label_encoding_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_dataframe_columns(
    df: pd.DataFrame, required_columns: set[str], df_name: str
) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def validate_labels_in_mapping(
    df: pd.DataFrame,
    label_col: str,
    code_to_idx: dict,
    df_name: str,
) -> None:
    labels_as_str = df[label_col].astype(str)
    unknown_labels = sorted(set(labels_as_str) - set(code_to_idx.keys()))
    if unknown_labels:
        raise ValueError(
            f"{df_name} contains labels not present in the predefined label encoding: "
            f"{unknown_labels[:10]}" + (" ..." if len(unknown_labels) > 10 else "")
        )


def save_split_ids(split_ids: pd.Index, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df = pd.DataFrame({"row_id": split_ids.astype(int)})
    split_df.to_csv(output_path, index=False)


def load_split_ids(split_path: str | Path) -> pd.Index:
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split_df = pd.read_csv(split_path)
    if "row_id" not in split_df.columns:
        raise ValueError(f"Split file {split_path} must contain a 'row_id' column.")
    return pd.Index(split_df["row_id"].astype(int).tolist())


def _can_use_stratify(labels: pd.Series, min_count: int = 2) -> tuple[bool, list[str]]:
    class_counts = labels.astype(str).value_counts()
    too_small_classes = class_counts[class_counts < min_count]
    return len(too_small_classes) == 0, too_small_classes.index.tolist()


def load_or_create_splits(
    df: pd.DataFrame,
    label_col: str,
    split_ids_dir: str | Path,
    seed: int,
    force_new_split: bool,
    val_size: float,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    split_ids_dir = Path(split_ids_dir)
    split_ids_dir.mkdir(parents=True, exist_ok=True)

    train_ids_path = split_ids_dir / "train_ids.csv"
    val_ids_path = split_ids_dir / "val_ids.csv"
    test_ids_path = split_ids_dir / "test_ids.csv"

    split_files_exist = (
        train_ids_path.exists() and val_ids_path.exists() and test_ids_path.exists()
    )

    if split_files_exist and not force_new_split:
        print(f"Loading existing splits from {split_ids_dir}")
        train_ids = load_split_ids(train_ids_path)
        val_ids = load_split_ids(val_ids_path)
        test_ids = load_split_ids(test_ids_path)
        return df.loc[train_ids].copy(), df.loc[val_ids].copy(), df.loc[test_ids].copy()

    labels = df[label_col].astype(str)
    use_stratify_first, small_classes_first = _can_use_stratify(labels, min_count=2)
    stratify_labels_first = labels if use_stratify_first else None

    train_ids, temp_ids = train_test_split(
        df.index,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=stratify_labels_first,
    )

    temp_df = df.loc[temp_ids].copy()
    temp_labels = temp_df[label_col].astype(str)
    relative_test_size = test_size / (val_size + test_size)
    use_stratify_second, _ = _can_use_stratify(temp_labels, min_count=2)
    stratify_labels_second = temp_labels if use_stratify_second else None

    val_ids, test_ids = train_test_split(
        temp_df.index,
        test_size=relative_test_size,
        random_state=seed,
        stratify=stratify_labels_second,
    )

    save_split_ids(pd.Index(train_ids), train_ids_path)
    save_split_ids(pd.Index(val_ids), val_ids_path)
    save_split_ids(pd.Index(test_ids), test_ids_path)

    return df.loc[train_ids].copy(), df.loc[val_ids].copy(), df.loc[test_ids].copy()


def _get_git_info() -> dict:
    commit = os.environ.get("GIT_COMMIT")
    branch = os.environ.get("GIT_BRANCH")

    if not commit or commit == "unknown":
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            commit = "unknown"
            branch = "unknown"

    return {
        "pre_training_git_commit": commit,
        "pre_training_git_branch": branch,
        "pre_training_timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def run_text_training(
    processed_data_dir: str | Path = "data/processed",
    train_config_path: str | Path = "configs/text_train_config.yaml",
    preprocessing_config_path: str | Path = "configs/text_preprocessing_config.yaml",
    model_save_path: str | Path = "models/best_text_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
) -> tuple[dict, dict]:
    processed_data_dir = Path(processed_data_dir)

    train_csv = processed_data_dir / "train.csv"
    val_csv = processed_data_dir / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Processed train data not found: {train_csv}. Run preprocess-text first."
        )
    if not val_csv.exists():
        raise FileNotFoundError(
            f"Processed val data not found: {val_csv}. Run preprocess-text first."
        )

    config = load_config(train_config_path)
    label_encoding = load_label_encoding(label_encoding_path)

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))
    seed = int(training_config.get("seed", 42))
    subset = int(training_config.get("subset", 0))
    lr = float(training_config.get("learning_rate", 2e-5))
    epochs = int(training_config.get("num_epochs", 5))
    wd = float(training_config.get("weight_decay", 0.01))

    model_name = model_config.get("name", "bert-base-multilingual-cased")
    pretrained = bool(model_config.get("pretrained", True))
    freeze_backbone = bool(model_config.get("freeze_backbone", False))

    label_col = data_config.get("label_col", "prdtypecode")
    designation_col = data_config.get("designation_col", "designation")
    description_col = data_config.get("description_col", "description")
    return_quality_report = bool(data_config.get("return_quality_report", False))

    set_seed(seed)

    # Load preprocessed splits directly
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if subset > 0:
        train_df = train_df[:subset]
        val_df = val_df[: int(subset * 0.2)]

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Train or val split is empty.")

    num_classes = len(label_encoding["classes"])

    train_dataset = RakutenTextDataset(
        dataframe=train_df,
        label_encoding=label_encoding,
        config_path=preprocessing_config_path,
        designation_col=designation_col,
        description_col=description_col,
        label_col=label_col,
        return_quality_report=return_quality_report,
    )
    val_dataset = RakutenTextDataset(
        dataframe=val_df,
        label_encoding=label_encoding,
        config_path=preprocessing_config_path,
        designation_col=designation_col,
        description_col=description_col,
        label_col=label_col,
        return_quality_report=return_quality_report,
    )

    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_text_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    git_info = _get_git_info()

    with mlflow.start_run(run_name="text-classifier-training"):
        mlflow.log_params(git_info)
        mlflow.log_params(
            {
                "processed_data_dir": str(processed_data_dir),
                "train_size": len(train_df),
                "val_size": len(val_df),
                "num_classes": num_classes,
            }
        )
        mlflow.log_params(
            {
                "model_name": model_name,
                "pretrained": pretrained,
                "freeze_backbone": freeze_backbone,
            }
        )
        mlflow.log_params(
            {
                "learning_rate": lr,
                "num_epochs": epochs,
                "weight_decay": wd,
                "batch_size": batch_size,
                "seed": seed,
                "subset": subset if subset > 0 else "full",
                "optimizer": "AdamW",
            }
        )
        mlflow.log_param("model_output_path", str(model_save_path))
        mlflow.log_artifact(str(train_config_path), artifact_path="configs")
        mlflow.log_artifact(str(preprocessing_config_path), artifact_path="configs")

        trained_model, history = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config_path=train_config_path,
            num_classes=num_classes,
            model_save_path=model_save_path,
        )

        mlflow.log_metrics(
            {
                "final_best_val_macro_f1": max(history["val_macro_f1"]),
                "final_best_val_accuracy": max(history["val_accuracy"]),
                "final_best_val_loss": min(history["val_loss"]),
            }
        )
        mlflow.pytorch.log_model(
            pytorch_model=trained_model,
            artifact_path="model",
            registered_model_name="text-classifier",
        )
        run_id = mlflow.active_run().info.run_id
        run_id_path = Path("results/mlflow_run_id.txt")
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(run_id)
        print(f"Saved MLflow run ID {run_id} to {run_id_path}")

    metrics_path = Path("results/dvc_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "val_macro_f1": max(history["val_macro_f1"]),
                "val_accuracy": max(history["val_accuracy"]),
                "val_loss": min(history["val_loss"]),
            },
            indent=2,
        )
    )

    return history, label_encoding


if __name__ == "__main__":
    history, label_encoding = run_text_training(
        processed_data_dir="data/processed",
        train_config_path="configs/text_train_config.yaml",
        preprocessing_config_path="configs/text_preprocessing_config.yaml",
        model_save_path="models/best_text_model.pt",
        label_encoding_path="configs/label_encoding.json",
    )
    print("Training finished.")
    print(f"Number of classes: {len(label_encoding['classes'])}")
    print(f"Best val macro-F1: {max(history['val_macro_f1']):.4f}")
