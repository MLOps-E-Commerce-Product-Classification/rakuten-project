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


def validate_labels_in_mapping(
    df: pd.DataFrame, label_col: str, code_to_idx: dict, df_name: str
) -> None:
    labels_as_str = df[label_col].astype(str)
    unknown_labels = sorted(set(labels_as_str) - set(code_to_idx.keys()))
    if unknown_labels:
        raise ValueError(
            f"{df_name} contains labels not present in the predefined label encoding: {unknown_labels[:10]}"
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
    return pd.Index(split_df["row_id"].astype(int).tolist())


def _can_use_stratify(labels: pd.Series, min_count: int = 2) -> tuple[bool, list[str]]:
    class_counts = labels.astype(str).value_counts()
    too_small_classes = class_counts[class_counts < min_count]
    return len(too_small_classes) == 0, too_small_classes.index.tolist()


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
    batch_size, seed, subset = (
        int(training_config.get("batch_size", 32)),
        int(training_config.get("seed", 42)),
        int(training_config.get("subset", 0)),
    )
    lr, epochs = (
        float(training_config.get("learning_rate", 2e-5)),
        int(training_config.get("num_epochs", 5)),
    )

    model_name = model_config.get("name", "bert-base-multilingual-cased")
    pretrained, freeze_backbone = (
        bool(model_config.get("pretrained", True)),
        bool(model_config.get("freeze_backbone", False)),
    )
    label_col, designation_col, description_col = (
        data_config.get("label_col", "prdtypecode"),
        data_config.get("designation_col", "designation"),
        data_config.get("description_col", "description"),
    )

    set_seed(seed)
    train_df, val_df = pd.read_csv(train_csv), pd.read_csv(val_csv)
    if subset > 0:
        train_df, val_df = train_df[:subset], val_df[: int(subset * 0.2)]

    num_classes = len(label_encoding["classes"])
    train_dataset = RakutenTextDataset(
        train_df,
        label_encoding,
        preprocessing_config_path,
        designation_col,
        description_col,
        label_col,
    )
    val_dataset = RakutenTextDataset(
        val_df,
        label_encoding,
        preprocessing_config_path,
        designation_col,
        description_col,
        label_col,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
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
        mlflow.log_params({"model_name": model_name, "epochs": epochs, "lr": lr})

        trained_model, history = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config_path=train_config_path,
            num_classes=num_classes,
            model_save_path=model_save_path,
        )

        mlflow.log_metrics({"final_best_val_macro_f1": max(history["val_macro_f1"])})
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

        run_id = mlflow.active_run().info.run_id
        Path("results/mlflow_run_id.txt").write_text(run_id)

    return history, label_encoding


if __name__ == "__main__":
    run_text_training()
