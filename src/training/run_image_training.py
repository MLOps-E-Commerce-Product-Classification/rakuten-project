from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.data.image_dataset import RakutenImageDataset
from src.models.image_classifier import build_image_model
from src.training.train_image import train_model


ARTIFACTS_PATH = Path("artifacts")
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def fit_label_encoder(train_labels: pd.Series) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(train_labels.astype(str))
    return encoder


def apply_label_encoding(
    df: pd.DataFrame,
    label_col: str,
    encoder: LabelEncoder,
    encoded_label_col: str = "label",
) -> pd.DataFrame:
    df = df.copy()
    df[encoded_label_col] = encoder.transform(df[label_col].astype(str))
    return df


def save_label_mapping(
    encoder: LabelEncoder,
    output_path: str | Path = ARTIFACTS_PATH / "label_mapping.json",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping = {
        str(original_label): int(encoded_label)
        for encoded_label, original_label in enumerate(encoder.classes_)
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def run_image_training(
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    train_image_dir: str | Path,
    val_image_dir: str | Path | None = None,
    train_config_path: str | Path = "configs/train_config.yaml",
    preprocessing_config_path: str | Path = "configs/image_preprocessing_config.yaml",
    model_save_path: str | Path = "models/best_image_model.pt",
    label_mapping_path: str | Path = ARTIFACTS_PATH / "label_mapping.json",
) -> tuple[torch.nn.Module, dict, LabelEncoder]:
    """
    Run end-to-end image training with robust handling for:
    - YAML configs
    - Rakuten label codes via label encoding
    - flexible local / mounted drive paths

    Parameters
    ----------
    train_csv_path : str | Path
        Path to training split CSV.
    val_csv_path : str | Path
        Path to validation split CSV.
    train_image_dir : str | Path
        Directory containing training images.
    val_image_dir : str | Path | None
        Directory containing validation images.
        If None, train_image_dir is used.
    train_config_path : str | Path
        Path to training YAML config.
    preprocessing_config_path : str | Path
        Path to preprocessing YAML config.
    model_save_path : str | Path
        Where to save best model weights.
    label_mapping_path : str | Path
        Where to save the label mapping JSON.

    Returns
    -------
    tuple[torch.nn.Module, dict, LabelEncoder]
        trained_model, history, label_encoder
    """
    config = load_config(train_config_path)

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
    encoded_label_col = data_config.get("encoded_label_col", "label")
    return_quality_report = bool(data_config.get("return_quality_report", False))

    set_seed(seed)

    train_csv_path = Path(train_csv_path)
    val_csv_path = Path(val_csv_path)
    train_image_dir = Path(train_image_dir)
    val_image_dir = Path(val_image_dir) if val_image_dir is not None else train_image_dir
    model_save_path = Path(model_save_path)
    label_mapping_path = Path(label_mapping_path)

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

    # ----------------------------------------------------
    # Label encoding for Rakuten codes
    # ----------------------------------------------------
    label_encoder = fit_label_encoder(train_df[label_col])

    train_df = apply_label_encoding(
        train_df,
        label_col=label_col,
        encoder=label_encoder,
        encoded_label_col=encoded_label_col,
    )

    val_labels_as_str = val_df[label_col].astype(str)
    unseen_val_labels = sorted(set(val_labels_as_str) - set(label_encoder.classes_))
    if unseen_val_labels:
        raise ValueError(
            "Validation set contains labels not present in the training set: "
            f"{unseen_val_labels[:10]}"
            + (" ..." if len(unseen_val_labels) > 10 else "")
        )

    val_df = apply_label_encoding(
        val_df,
        label_col=label_col,
        encoder=label_encoder,
        encoded_label_col=encoded_label_col,
    )

    save_label_mapping(label_encoder, label_mapping_path)

    num_classes = len(label_encoder.classes_)

    train_dataset = RakutenImageDataset(
        dataframe=train_df,
        image_dir=train_image_dir,
        config_path=preprocessing_config_path,
        image_id_col=image_id_col,
        label_col=encoded_label_col,
        return_quality_report=return_quality_report,
    )

    val_dataset = RakutenImageDataset(
        dataframe=val_df,
        image_dir=val_image_dir,
        config_path=preprocessing_config_path,
        image_id_col=image_id_col,
        label_col=encoded_label_col,
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

    return trained_model, history, label_encoder


if __name__ == "__main__":
    trained_model, history, label_encoder = run_image_training(
        train_csv_path="data/train_split.csv",
        val_csv_path="data/val_split.csv",
        train_image_dir="data/images",
        val_image_dir="data/images",
        train_config_path="configs/train_config.yaml",
        preprocessing_config_path="configs/image_preprocessing_config.yaml",
        model_save_path="models/best_image_model.pt",
        label_mapping_path="artifacts/label_mapping.json",
    )

    print("Training finished.")
    print("Best model saved to: models/best_image_model.pt")
    print("Label mapping saved to: artifacts/label_mapping.json")
    print(f"Number of classes: {len(label_encoder.classes_)}")