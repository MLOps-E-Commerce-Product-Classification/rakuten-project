from pathlib import Path
import json
import os
import subprocess
import shutil
from datetime import datetime

import mlflow
import mlflow.pytorch
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import torch
import yaml
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data.text_dataset import RakutenTextDataset
from src.models.text_classifier import build_text_model
from src.training.train_text import train_model

load_dotenv()

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MODEL_NAME = "text-classifier"
SELECTION_METRIC = "final_best_val_macro_f1"
IMPROVEMENT_THRESHOLD = 0.005  # = 0.5 percentage points

# ---------------------------------------------------------------------
# MLflow Helpers (Promotion & Git)
# ---------------------------------------------------------------------
def get_production_metric(model_name: str, metric_name: str) -> float:
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise RuntimeError(f"No Production model found for {model_name}")
    run = client.get_run(versions[0].run_id)
    if metric_name not in run.data.metrics:
        raise RuntimeError(f"Production model missing metric '{metric_name}'")
    return run.data.metrics[metric_name]

def get_candidate_model_version(model_name: str, run_id: str):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.run_id == run_id:
            return v
    raise RuntimeError("Could not resolve model version for current run")

def get_git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        commit, branch = "unknown", "unknown"
    return {
        "git_commit": commit,
        "git_branch": branch,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

# ---------------------------------------------------------------------
# PyFunc wrapper (Wichtig für BentoML & API Inferenz)
# ---------------------------------------------------------------------
class RakutenModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import AutoTokenizer
        self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])
        self.model.eval()
        # Nutzt jetzt den dynamischen Tokenizer-Namen aus den Artifacts
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["tokenizer_name"])

    def predict(self, context, model_input):
        import torch
        # Optimierte String-Konkatenierung aus dem API-Commit
        texts = (
            model_input["designation"].astype(str)
            + " "
            + model_input["description"].astype(str)
        ).tolist()

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        return probs.numpy()

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_label_encoding(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------
# Main training + promotion logic
# ---------------------------------------------------------------------
def run_text_training(
    processed_data_dir="data/processed",
    train_config_path="configs/text_train_config.yaml",
    preprocessing_config_path="configs/text_preprocessing_config.yaml",
    label_encoding_path="configs/label_encoding.json",
    retrain=False,
    base_model_uri=None,
):
    processed_data_dir = Path(processed_data_dir)
    train_df = pd.read_csv(processed_data_dir / "train.csv")
    val_df = pd.read_csv(processed_data_dir / "val.csv")

    if train_df.empty or val_df.empty:
        raise RuntimeError("Train or validation split is empty")

    config = load_config(train_config_path)
    label_encoding = load_label_encoding(label_encoding_path)

    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]

    set_seed(training_cfg.get("seed", 42))

    train_ds = RakutenTextDataset(
        train_df, label_encoding, preprocessing_config_path,
        data_cfg["designation_col"], data_cfg["description_col"], data_cfg["label_col"]
    )
    val_ds = RakutenTextDataset(
        val_df, label_encoding, preprocessing_config_path,
        data_cfg["designation_col"], data_cfg["description_col"], data_cfg["label_col"]
    )

    train_dl = DataLoader(train_ds, batch_size=training_cfg["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=training_cfg["batch_size"], shuffle=False)

    if retrain:
        if not base_model_uri:
            raise ValueError("Retrain requested but base_model_uri is missing")
        model = mlflow.pytorch.load_model(base_model_uri)
    else:
        model = build_text_model(
            model_cfg["name"], len(label_encoding["classes"]),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False)
        )

    with mlflow.start_run(run_name="text-training"):
        mlflow.log_params(get_git_info())

        trained_model, history = train_model(
            model, train_dl, val_dl, train_config_path,
            num_classes=len(label_encoding["classes"])
        )

        candidate_metric = max(history["val_macro_f1"])
        mlflow.log_metrics({
            SELECTION_METRIC: candidate_metric,
            "final_best_val_accuracy": max(history["val_accuracy"]),
            "final_best_val_loss": min(history["val_loss"]),
        })

        # Vorbereitung für PyFunc Logging
        trained_model.to("cpu")
        tmp = Path("models/_tmp_pytorch")
        if tmp.exists(): shutil.rmtree(tmp)
        mlflow.pytorch.save_model(trained_model, tmp)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RakutenModelWrapper(),
            artifacts={
                "pytorch_model": str(tmp),
                "tokenizer_name": model_cfg["name"],
            },
            registered_model_name=MODEL_NAME,
            pip_requirements=["torch", "transformers", "pandas", "numpy"],
        )

        shutil.rmtree(tmp)
        run_id = mlflow.active_run().info.run_id

    # Promotion Decision
    client = MlflowClient()
    candidate_version = get_candidate_model_version(MODEL_NAME, run_id)

    try:
        prod_metric = get_production_metric(MODEL_NAME, SELECTION_METRIC)
        improvement = candidate_metric - prod_metric
    except RuntimeError:
        prod_metric, improvement = None, float("inf")
        print("No Production model found (initial deployment)")

    print(f"Candidate {SELECTION_METRIC}: {candidate_metric:.4f}")
    if prod_metric: print(f"Production {SELECTION_METRIC}: {prod_metric:.4f}")
    print(f"Improvement: {improvement:.4f}")

    if improvement >= IMPROVEMENT_THRESHOLD:
        print("✅ Promoting model to Production")
        for v in client.get_latest_versions(MODEL_NAME, stages=["Production"]):
            client.transition_model_version_stage(MODEL_NAME, v.version, stage="Archived")
        client.transition_model_version_stage(MODEL_NAME, candidate_version.version, stage="Production")
    else:
        print("❌ Promotion skipped (threshold not met)")

    return history

if __name__ == "__main__":
    run_text_training()
