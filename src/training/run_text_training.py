from pathlib import Path
import json
import os
import shutil
from datetime import datetime

import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data.text_dataset import RakutenTextDataset
from src.models.text_classifier import build_text_model
from src.training.train_text import train_model

load_dotenv()


class RakutenModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import AutoTokenizer
        import mlflow.pytorch

        self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def predict(self, context, model_input):
        import torch

        designations = model_input["designation"].astype(str).tolist()
        descriptions = model_input["description"].astype(str).tolist()

        texts = [f"{desig} {desc}" for desig, desc in zip(designations, descriptions)]

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs.numpy()


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

    if (
        train_ids_path.exists()
        and val_ids_path.exists()
        and test_ids_path.exists()
        and not force_new_split
    ):
        return (
            df.loc[load_split_ids(train_ids_path)].copy(),
            df.loc[load_split_ids(val_ids_path)].copy(),
            df.loc[load_split_ids(test_ids_path)].copy(),
        )

    labels = df[label_col].astype(str)
    use_stratify, _ = _can_use_stratify(labels)
    train_ids, temp_ids = train_test_split(
        df.index,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=labels if use_stratify else None,
    )

    temp_df = df.loc[temp_ids]
    temp_labels = temp_df[label_col].astype(str)
    use_stratify_val, _ = _can_use_stratify(temp_labels)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        stratify=temp_labels if use_stratify_val else None,
    )

    save_split_ids(pd.Index(train_ids), train_ids_path)
    save_split_ids(pd.Index(val_ids), val_ids_path)
    save_split_ids(pd.Index(test_ids), test_ids_path)
    return df.loc[train_ids].copy(), df.loc[val_ids].copy(), df.loc[test_ids].copy()


def _get_git_info() -> dict:
    commit = os.environ.get("GIT_COMMIT", "unknown")
    branch = os.environ.get("GIT_BRANCH", "unknown")
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
    train_csv, val_csv = (
        processed_data_dir / "train.csv",
        processed_data_dir / "val.csv",
    )
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Processed data not found.")

    config = load_config(train_config_path)
    label_encoding = load_label_encoding(label_encoding_path)
    training_config, model_config, data_config = (
        config.get("training", {}),
        config.get("model", {}),
        config.get("data", {}),
    )

    set_seed(int(training_config.get("seed", 42)))
    train_df, val_df = pd.read_csv(train_csv), pd.read_csv(val_csv)
    subset = int(training_config.get("subset", 0))
    if subset > 0:
        train_df, val_df = train_df[:subset], val_df[: int(subset * 0.2)]

    num_classes = len(label_encoding["classes"])
    train_dataset = RakutenTextDataset(
        train_df,
        label_encoding,
        preprocessing_config_path,
        data_config.get("designation_col", "designation"),
        data_config.get("description_col", "description"),
        data_config.get("label_col", "prdtypecode"),
    )
    val_dataset = RakutenTextDataset(
        val_df,
        label_encoding,
        preprocessing_config_path,
        data_config.get("designation_col", "designation"),
        data_config.get("description_col", "description"),
        data_config.get("label_col", "prdtypecode"),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_text_model(
        model_name=model_config.get("name", "bert-base-multilingual-cased"),
        num_classes=num_classes,
        pretrained=bool(model_config.get("pretrained", True)),
        freeze_backbone=bool(model_config.get("freeze_backbone", False)),
    )

    with mlflow.start_run(run_name="text-classifier-training"):
        mlflow.log_params(_get_git_info())
        trained_model, history = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config_path=train_config_path,
            num_classes=num_classes,
            model_save_path=model_save_path,
        )
        trained_model.to("cpu")

        temp_pytorch_dir = Path("models/temp_mlflow_pytorch")
        if temp_pytorch_dir.exists():
            shutil.rmtree(temp_pytorch_dir)
        mlflow.pytorch.save_model(trained_model, temp_pytorch_dir)

        mlflow.pyfunc.log_model(
            artifact_path="mlflow_model",
            python_model=RakutenModelWrapper(),
            artifacts={"pytorch_model": str(temp_pytorch_dir)},
            registered_model_name="text-classifier",
            pip_requirements=["torch", "transformers", "pandas", "numpy"],
        )
        if temp_pytorch_dir.exists():
            shutil.rmtree(temp_pytorch_dir)
        Path("results/mlflow_run_id.txt").write_text(mlflow.active_run().info.run_id)

    return history, label_encoding


if __name__ == "__main__":
    run_text_training()
