from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
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
    """
    Check whether stratified splitting is possible.

    Returns
    -------
    tuple[bool, list[str]]
        (can_use_stratify, classes_with_too_few_samples)
    """
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
    """
    Create or load train/val/test splits based on row ids.

    The split files are:
    - train_ids.csv
    - val_ids.csv
    - test_ids.csv

    Split strategy:
    - Try stratified split if possible
    - If some classes are too small, fall back to random split without stratification
    """
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

        train_df = df.loc[train_ids].copy()
        val_df = df.loc[val_ids].copy()
        test_df = df.loc[test_ids].copy()

        return train_df, val_df, test_df

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")

    labels = df[label_col].astype(str)

    # ----------------------------------------------------
    # First split: train vs temp(val+test)
    # ----------------------------------------------------
    use_stratify_first, small_classes_first = _can_use_stratify(labels, min_count=2)

    if use_stratify_first:
        stratify_labels_first = labels
        print("Using stratified split for train vs. temp split.")
    else:
        stratify_labels_first = None
        print(
            "Warning: Falling back to non-stratified split for train vs. temp split "
            "because some classes have fewer than 2 samples: "
            f"{small_classes_first}"
        )

    train_ids, temp_ids = train_test_split(
        df.index,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=stratify_labels_first,
    )

    temp_df = df.loc[temp_ids].copy()
    temp_labels = temp_df[label_col].astype(str)

    # ----------------------------------------------------
    # Second split: val vs test
    # ----------------------------------------------------
    relative_test_size = test_size / (val_size + test_size)

    use_stratify_second, small_classes_second = _can_use_stratify(
        temp_labels, min_count=2
    )

    if use_stratify_second:
        stratify_labels_second = temp_labels
        print("Using stratified split for val vs. test split.")
    else:
        stratify_labels_second = None
        print(
            "Warning: Falling back to non-stratified split for val vs. test split "
            "because some classes have fewer than 2 samples in the temporary split: "
            f"{small_classes_second}"
        )

    val_ids, test_ids = train_test_split(
        temp_df.index,
        test_size=relative_test_size,
        random_state=seed,
        stratify=stratify_labels_second,
    )

    save_split_ids(pd.Index(train_ids), train_ids_path)
    save_split_ids(pd.Index(val_ids), val_ids_path)
    save_split_ids(pd.Index(test_ids), test_ids_path)

    print(f"Saved train/val/test split ids to {split_ids_dir}")

    train_df = df.loc[train_ids].copy()
    val_df = df.loc[val_ids].copy()
    test_df = df.loc[test_ids].copy()

    return train_df, val_df, test_df


def run_image_training(
    x_data_csv_path: str | Path,
    y_data_csv_path: str | Path,
    image_dir: str | Path,
    split_ids_dir: str | Path,
    force_new_split: bool = False,
    train_config_path: str | Path = "configs/image_train_config.yaml",
    preprocessing_config_path: str | Path = "configs/image_preprocessing_config.yaml",
    model_save_path: str | Path = "models/best_image_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
    use_best_config_if_available: bool = True,   # <-- NEU
) -> tuple[torch.nn.Module, dict, dict]:
    """
    Run end-to-end image training with:
    - X/Y CSV loading
    - predefined label encoding
    - train/val/test splitting with saved split ids
    - train on train split
    - early stopping / validation on val split
    - keep test split for later evaluation

    Returns
    -------
    tuple[torch.nn.Module, dict, dict]
        trained_model, history, label_encoding
    """

    train_config_path = Path(train_config_path)

    best_config_path = train_config_path.parent / "best_train_config.yaml"

    if use_best_config_if_available and best_config_path.exists():
        print(f"Using best training config from random search: {best_config_path}")
        train_config_path = best_config_path
    else:
        print(f"Using training config: {train_config_path}")

    config = load_config(train_config_path)
    label_encoding = load_label_encoding(label_encoding_path)

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    split_config = config.get("split", {})

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))
    seed = int(training_config.get("seed", 42))
    subset = int(training_config.get("subset", 0))

    model_name = model_config.get("name", "efficientnet_b0")
    pretrained = bool(model_config.get("pretrained", True))
    freeze_backbone = bool(model_config.get("freeze_backbone", False))

    image_id_col = data_config.get("image_id_col", "imageid")
    product_id_col = data_config.get("product_id_col", "productid")
    label_col = data_config.get("label_col", "prdtypecode")
    return_quality_report = bool(data_config.get("return_quality_report", False))

    val_size = float(split_config.get("val_size", 0.1))
    test_size = float(split_config.get("test_size", 0.1))

    set_seed(seed)

    x_data_csv_path = Path(x_data_csv_path)
    y_data_csv_path = Path(y_data_csv_path)
    image_dir = Path(image_dir)
    split_ids_dir = Path(split_ids_dir)
    model_save_path = Path(model_save_path)
    label_encoding_path = Path(label_encoding_path)

    if not x_data_csv_path.exists():
        raise FileNotFoundError(f"X data CSV not found: {x_data_csv_path}")
    if not y_data_csv_path.exists():
        raise FileNotFoundError(f"Y data CSV not found: {y_data_csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    x_df = pd.read_csv(x_data_csv_path)
    y_df = pd.read_csv(y_data_csv_path)


    if len(x_df) != len(y_df):
        raise ValueError(
            f"X and Y CSVs must have the same number of rows, got {len(x_df)} and {len(y_df)}."
        )

    df = pd.concat([x_df, y_df], axis=1)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    required_columns = {image_id_col, product_id_col, label_col}
    validate_dataframe_columns(df, required_columns, "Merged training data")

    code_to_idx = label_encoding["code_to_idx"]
    validate_labels_in_mapping(
        df,
        label_col=label_col,
        code_to_idx=code_to_idx,
        df_name="Merged training data",
    )

    train_df, val_df, test_df = load_or_create_splits(
        df=df,
        label_col=label_col,
        split_ids_dir=split_ids_dir,
        seed=seed,
        force_new_split=force_new_split,
        val_size=val_size,
        test_size=test_size,
    )

    if subset > 0:
        train_df = train_df[:subset]
        val_df = val_df[:int(subset*0.2)]
        

    # Optional sanity check
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("At least one split is empty. Please check split sizes.")

    num_classes = len(label_encoding["classes"])

    train_dataset = RakutenImageDataset(
        dataframe=train_df,
        image_dir=image_dir,
        config_path=preprocessing_config_path,
        image_id_col=image_id_col,
        product_id_col=product_id_col,
        label_col=label_col,
        return_quality_report=return_quality_report,
        label_encoding_path=label_encoding_path,
    )

    val_dataset = RakutenImageDataset(
        dataframe=val_df,
        image_dir=image_dir,
        config_path=preprocessing_config_path,
        image_id_col=image_id_col,
        product_id_col=product_id_col,
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

    history["split_sizes"] = {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }

    return trained_model, history, label_encoding


if __name__ == "__main__":
    trained_model, history, label_encoding = run_image_training(
        x_data_csv_path="data/X_train_update.csv",
        y_data_csv_path="data/Y_train_CVw08PX.csv",
        image_dir="data/images/image_train",
        split_ids_dir="artifacts/splits",
        force_new_split=False,
        train_config_path="configs/image_train_config.yaml",
        preprocessing_config_path="configs/image_preprocessing_config.yaml",
        model_save_path="models/best_image_model.pt",
        label_encoding_path="configs/label_encoding.json",
    )

    print("Training finished.")
    print("Best model saved to: models/best_image_model.pt")
    print("Label encoding loaded from: configs/label_encoding.json")
    print(f"Number of classes: {len(label_encoding['classes'])}")
    print(f"Split sizes: {history['split_sizes']}")
