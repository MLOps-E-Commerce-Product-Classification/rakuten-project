from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.data.text_dataset import RakutenTextDataset
from src.models.text_classifier import build_text_model
from src.training.train_text import train_model


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


def validate_dataframe_columns(df: pd.DataFrame, required_columns: set[str], df_name: str) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def fit_label_encoder(train_labels: pd.Series) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(train_labels.astype(str))
    return encoder


def apply_label_encoding(df, label_col, encoder, encoded_label_col="label"):
    df = df.copy()
    df[encoded_label_col] = encoder.transform(df[label_col].astype(str))
    return df


def save_label_mapping(encoder: LabelEncoder, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {str(orig): int(enc) for enc, orig in enumerate(encoder.classes_)}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def run_text_training(
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    train_config_path: str | Path = "configs/text_train_config.yaml",
    preprocessing_config_path: str | Path = "configs/text_preprocessing_config.yaml",
    model_save_path: str | Path = "models/best_text_model.pt",
    label_mapping_path: str | Path = ARTIFACTS_PATH / "label_mapping.json",
) -> tuple[torch.nn.Module, dict, LabelEncoder]:
    config = load_config(train_config_path)

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))
    seed = int(training_config.get("seed", 42))

    model_name = model_config.get("name", "bert-base-multilingual-cased")
    pretrained = bool(model_config.get("pretrained", True))
    freeze_backbone = bool(model_config.get("freeze_backbone", False))

    label_col = data_config.get("label_col", "prdtypecode")
    encoded_label_col = data_config.get("encoded_label_col", "label")
    designation_col = data_config.get("designation_col", "designation")
    description_col = data_config.get("description_col", "description")
    return_quality_report = bool(data_config.get("return_quality_report", False))

    set_seed(seed)

    train_csv_path = Path(train_csv_path)
    val_csv_path = Path(val_csv_path)

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    validate_dataframe_columns(train_df, {designation_col, label_col}, "Training CSV")
    validate_dataframe_columns(val_df, {designation_col, label_col}, "Validation CSV")

    label_encoder = fit_label_encoder(train_df[label_col])

    train_df = apply_label_encoding(train_df, label_col, label_encoder, encoded_label_col)

    unseen = sorted(set(val_df[label_col].astype(str)) - set(label_encoder.classes_))
    if unseen:
        raise ValueError(f"Validation set contains unseen labels: {unseen[:10]}")

    val_df = apply_label_encoding(val_df, label_col, label_encoder, encoded_label_col)
    save_label_mapping(label_encoder, label_mapping_path)

    num_classes = len(label_encoder.classes_)

    train_dataset = RakutenTextDataset(
        dataframe=train_df,
        config_path=preprocessing_config_path,
        designation_col=designation_col,
        description_col=description_col,
        label_col=encoded_label_col,
        return_quality_report=return_quality_report,
    )

    val_dataset = RakutenTextDataset(
        dataframe=val_df,
        config_path=preprocessing_config_path,
        designation_col=designation_col,
        description_col=description_col,
        label_col=encoded_label_col,
        return_quality_report=return_quality_report,
    )

    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    model = build_text_model(
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
    trained_model, history, label_encoder = run_text_training(
        train_csv_path="data/train_split.csv",
        val_csv_path="data/val_split.csv",
    )

    print("Training finished.")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Best val macro-F1: {max(history['val_macro_f1']):.4f}")
