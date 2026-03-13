from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.image_dataset import RakutenImageDataset
from src.models.image_classifier import build_image_model
from src.training.train_image import train_model


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_encoding(label_encoding_path: str | Path) -> dict:
    label_encoding_path = Path(label_encoding_path)

    if not label_encoding_path.exists():
        raise FileNotFoundError(
            f"Label encoding file not found: {label_encoding_path}"
        )

    with label_encoding_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            f"{unknown_labels[:10]}"
            + (" ..." if len(unknown_labels) > 10 else "")
        )


def run_image_training(
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    train_image_dir: str | Path,
    val_image_dir: str | Path | None = None,
    train_config_path: str | Path = "configs/train_config.yaml",
    preprocessing_config_path: str | Path = "configs/image_preprocessing_config.yaml",
    model_save_path: str | Path = "models/best_image_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
) -> tuple[torch.nn.Module, dict, dict]:
    """
    Run end-to-end image training with:
    - YAML configs
    - predefined label encoding
    - flexible local / mounted drive paths

    Returns
    -------
    tuple[torch.nn.Module, dict, dict]
        trained_model, history, label_encoding
    """
    config = load_config(train_config_path)
    label_encoding = load_label_encoding(label_encoding_path)

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))
    seed = int(training_config.get("seed", 42))

    model_name = model_config.get("name", "efficientnet_b0")
    pretrained = bool(model_config.get("pretrained", True))
    freeze_backbone = bool(model_config.get("freeze_backbone", False))

    image_id_col = data_config.get("image_id_col", "image_id")
    label_col = data_config.get("label_col", "label")
    return_quality_report = bool(data_config.get("return_quality_report", False))

    set_seed(seed)

    train_csv_path = Path(train_csv_path)
    val_csv_path = Path(val_csv_path)
    train_image_dir = Path(train_image_dir)
    val_image_dir = Path(val_image_dir) if val_image_dir is not None else train_image_dir
    model_save_path = Path(model_save_path)
    label_encoding_path = Path(label_encoding_path)

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv_path}")
    if not train_image_dir.exists():
        raise FileNotFoundError(f"Training image directory not found: {train_image_dir}")
    if not val_image_dir.exists():
        raise FileNotFoundError(f"Validation image directory not found: {val_image_dir}")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    required_columns = {image_id_col, label_col}
    validate_dataframe_columns(train_df, required_columns, "Training CSV")
    validate_dataframe_columns(val_df, required_columns, "Validation CSV")

    code_to_idx = label_encoding["code_to_idx"]

    validate_labels_in_mapping(
        train_df,
        label_col=label_col,
        code_to_idx=code_to_idx,
        df_name="Training CSV",
    )
    validate_labels_in_mapping(
        val_df,
        label_col=label_col,
        code_to_idx=code_to_idx,
        df_name="Validation CSV",
    )

    num_classes = len(label_encoding["classes"])

    train_dataset = RakutenImageDataset(
        dataframe=train_df,
        image_dir=train_image_dir,
        config_path=preprocessing_config_path,
        image_id_col=image_id_col,
        label_col=label_col,
        return_quality_report=return_quality_report,
        label_encoding_path=label_encoding_path,
    )

    val_dataset = RakutenImageDataset(
        dataframe=val_df,
        image_dir=val_image_dir,
        config_path=preprocessing_config_path,
        image_id_col=image_id_col,
        label_col=label_col,
        return_quality_report=return_quality_report,
        label_encoding_path=label_encoding_path,
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

    model = build_image_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    trained_model, history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config_path=train_config_path,
        num_classes=num_classes,
        model_save_path=model_save_path,
    )

    return trained_model, history, label_encoding


if __name__ == "__main__":
    trained_model, history, label_encoding = run_image_training(
        train_csv_path="data/train_split.csv",
        val_csv_path="data/val_split.csv",
        train_image_dir="data/images",
        val_image_dir="data/images",
        train_config_path="configs/train_config.yaml",
        preprocessing_config_path="configs/image_preprocessing_config.yaml",
        model_save_path="models/best_image_model.pt",
        label_encoding_path="configs/label_encoding.json",
    )

    print("Training finished.")
    print("Best model saved to: models/best_image_model.pt")
    print("Label encoding loaded from: configs/label_encoding.json")
    print(f"Number of classes: {len(label_encoding['classes'])}")