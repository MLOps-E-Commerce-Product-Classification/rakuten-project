from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data.text_dataset import RakutenTextDataset
from src.training.run_text_training import (
    load_config,
    load_label_encoding,
    set_seed,
    _get_git_info,
)
from src.training.train_text import train_model

load_dotenv()


def load_finetuning_model_from_registry(
    model_name: str, alias: str = "production"
) -> torch.nn.Module:
    """
    Loads the current 'production' model from the MLflow Registry.
    """
    model_uri = f"models:/{model_name}@{alias}"
    print(f"--- Fetching base model from Registry: {model_uri} ---")

    pyfunc_model = mlflow.pyfunc.load_model(model_uri)

    pytorch_model = pyfunc_model.unwrap_python_model().model
    pytorch_model.train()

    print(f"Successfully loaded model: {model_name}@{alias}")
    return pytorch_model


def mix_datasets(
    old_train_df: pd.DataFrame,
    new_train_df: pd.DataFrame,
    new_data_ratio: float,
    seed: int,
) -> pd.DataFrame:

    if new_data_ratio <= 0.0:
        print("new_data_ratio=0 → nur neue Daten werden verwendet.")
        return new_train_df.reset_index(drop=True)

    if new_data_ratio >= 1.0:
        print("new_data_ratio=1 → nur alte Daten werden verwendet.")
        return old_train_df.reset_index(drop=True)

    n_new = len(new_train_df)
    n_old_needed = int(n_new * (1 - new_data_ratio) / new_data_ratio)
    n_old_sample = min(n_old_needed, len(old_train_df))

    old_sample = old_train_df.sample(n=n_old_sample, random_state=seed)

    mixed = pd.concat([new_train_df, old_sample], ignore_index=True).sample(
        frac=1, random_state=seed
    )

    print(
        f"Dataset mix: {len(new_train_df)} new + {n_old_sample} old = {len(mixed)} total"
    )
    return mixed.reset_index(drop=True)


def run_text_finetuning(
    processed_data_dir: str | Path = "data/processed",
    new_processed_data_dir: str | Path = "data/processed_new",
    finetune_config_path: str | Path = "configs/text_finetune_config.yaml",
    preprocessing_config_path: str | Path = "configs/text_preprocessing_config.yaml",
    model_save_path: str | Path = "models/best_text_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
) -> tuple[dict, dict]:

    processed_data_dir = Path(processed_data_dir)
    new_processed_data_dir = Path(new_processed_data_dir)

    config = load_config(finetune_config_path)
    label_encoding = load_label_encoding(label_encoding_path)

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))
    seed = int(training_config.get("seed", 42))
    subset = int(training_config.get("subset", 0))
    lr = float(training_config.get("learning_rate", 5e-6))
    epochs = int(training_config.get("num_epochs", 2))
    wd = float(training_config.get("weight_decay", 0.01))

    model_name = model_config.get("name", "bert-base-multilingual-cased")
    freeze_backbone = bool(model_config.get("freeze_backbone", False))

    model_name_registry = model_config.get("mlflow_model_name", "text-classifier")
    model_alias = model_config.get("mlflow_alias", "production")

    label_col = data_config.get("label_col", "prdtypecode")
    designation_col = data_config.get("designation_col", "designation")
    description_col = data_config.get("description_col", "description")
    return_quality_report = bool(data_config.get("return_quality_report", False))
    new_data_ratio = float(data_config.get("new_data_ratio", 0.3))

    set_seed(seed)

    # --- Load old data ---
    old_train_csv = processed_data_dir / "train.csv"
    val_csv = processed_data_dir / "val.csv"

    if not old_train_csv.exists():
        raise FileNotFoundError(f"Old train data not found: {old_train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Val data not found: {val_csv}")

    old_train_df = pd.read_csv(old_train_csv)
    val_df = pd.read_csv(val_csv)

    # --- Load new data ---
    new_train_csv = new_processed_data_dir / "train.csv"
    if new_train_csv.exists():
        new_train_df = pd.read_csv(new_train_csv)
        print(f"New data found: {len(new_train_df)} rows")
        train_df = mix_datasets(old_train_df, new_train_df, new_data_ratio, seed)
    else:
        print("No new data found → using old data only")
        train_df = old_train_df

    if subset > 0:
        train_df = train_df[:subset]
        val_df = val_df[: int(subset * 0.2)]

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Train or val split is empty.")

    num_classes = len(label_encoding["classes"])

    # --- Load model from MLflow ---
    model = load_finetuning_model_from_registry(
        model_name=model_name_registry, alias=model_alias
    )

    if freeze_backbone:
        print("Freezing BERT backbone")
        if hasattr(model, "bert"):
            for param in model.bert.parameters():
                param.requires_grad = False

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

    git_info = _get_git_info()

    with mlflow.start_run(run_name="text-classifier"):
        mlflow.log_params(git_info)

        mlflow.log_params(
            {
                "mode": "finetuning",
                "base_model_registry": f"{model_name_registry}@{model_alias}",
                "train_size": len(train_df),
                "val_size": len(val_df),
                "num_classes": num_classes,
                "new_data_ratio": new_data_ratio,
            }
        )

        mlflow.log_params(
            {
                "model_name": model_name,
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

        mlflow.log_artifact(str(finetune_config_path), artifact_path="configs")
        mlflow.log_artifact(str(preprocessing_config_path), artifact_path="configs")

        trained_model, history = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config_path=finetune_config_path,
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

        trained_model.to("cpu")

        temp_model_path = "models/temp_mlflow_pytorch_finetune"
        if os.path.exists(temp_model_path):
            shutil.rmtree(temp_model_path)

        mlflow.pytorch.save_model(trained_model, temp_model_path)

        from src.training.run_text_training import RakutenModelWrapper

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RakutenModelWrapper(),
            artifacts={"pytorch_model": temp_model_path},
            registered_model_name="text-classifier",
            pip_requirements=["torch", "transformers", "pandas", "numpy"],
        )

        shutil.rmtree(temp_model_path, ignore_errors=True)

        run_id = mlflow.active_run().info.run_id
        Path("results/mlflow_run_id_finetune.txt").write_text(run_id)

    metrics_path = Path("results/dvc_metrics_finetune.json")
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
