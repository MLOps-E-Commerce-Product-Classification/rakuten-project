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

MODEL_NAME = "text-classifier"
SELECTION_METRIC = "final_best_val_macro_f1"
IMPROVEMENT_THRESHOLD = 0.005  # = 0.5 percentage points

# ---------------------------------------------------------------------
# MLflow Pyfunc Wrapper (Wichtig für BentoML!)
# ---------------------------------------------------------------------
class RakutenModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import AutoTokenizer
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

# ---------------------------------------------------------------------
# MLflow Helpers (Promotion Logic)
# ---------------------------------------------------------------------
def get_production_metric(model_name: str, metric_name: str) -> float:
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise RuntimeError(f"No Production model found for {model_name}")
    run = client.get_run(versions[0].run_id)
    return run.data.metrics[metric_name]

def get_candidate_model_version(model_name: str, run_id: str):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.run_id == run_id:
            return v
    raise RuntimeError("Could not find model version for current run")

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _get_git_info() -> dict:
    commit = os.environ.get("GIT_COMMIT", "unknown")
    branch = os.environ.get("GIT_BRANCH", "unknown")
    if commit == "unknown":
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        except: pass
    return {
        "pre_training_git_commit": commit,
        "pre_training_git_branch": branch,
        "pre_training_timestamp": datetime.utcnow().isoformat() + "Z",
    }

# ---------------------------------------------------------------------
# Main training + promotion logic
# ---------------------------------------------------------------------
def run_text_training(
    processed_data_dir: str | Path = "data/processed",
    train_config_path: str | Path = "configs/text_train_config.yaml",
    preprocessing_config_path: str | Path = "configs/text_preprocessing_config.yaml",
    model_save_path: str | Path = "models/best_text_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
    retrain: bool = False,
    base_model_uri: str | None = None,
):
    processed_data_dir = Path(processed_data_dir)
    train_df = pd.read_csv(processed_data_dir / "train.csv")
    val_df = pd.read_csv(processed_data_dir / "val.csv")

    config = load_config(train_config_path)
    label_encoding = load_label_encoding(label_encoding_path)

    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]

    set_seed(training_cfg.get("seed", 42))

    train_dataset = RakutenTextDataset(train_df, label_encoding, preprocessing_config_path, 
                                       data_cfg["designation_col"], data_cfg["description_col"], data_cfg["label_col"])
    val_dataset = RakutenTextDataset(val_df, label_encoding, preprocessing_config_path, 
                                     data_cfg["designation_col"], data_cfg["description_col"], data_cfg["label_col"])

    train_dl = DataLoader(train_dataset, batch_size=training_cfg["batch_size"], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=training_cfg["batch_size"], shuffle=False)

    if retrain and base_model_uri:
        model = mlflow.pytorch.load_model(base_model_uri)
    else:
        model = build_text_model(model_cfg["name"], len(label_encoding["classes"]), 
                                 model_cfg.get("pretrained", True), model_cfg.get("freeze_backbone", False))

    with mlflow.start_run(run_name="text-classifier-training"):
        mlflow.log_params(_get_git_info())
        trained_model, history = train_model(model, train_dl, val_dl, train_config_path, 
                                             len(label_encoding["classes"]), model_save_path)

        candidate_metric = max(history["val_macro_f1"])
        mlflow.log_metrics({
            SELECTION_METRIC: candidate_metric,
            "final_best_val_accuracy": max(history["val_accuracy"]),
            "final_best_val_loss": min(history["val_loss"]),
        })

        # WICHTIG: Als Pyfunc loggen für BentoML Kompatibilität
        trained_model.to("cpu")
        temp_model_path = "models/temp_mlflow_pytorch"
        if os.path.exists(temp_model_path): shutil.rmtree(temp_model_path)
        mlflow.pytorch.save_model(trained_model, temp_model_path)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RakutenModelWrapper(),
            artifacts={"pytorch_model": temp_model_path},
            registered_model_name=MODEL_NAME,
            pip_requirements=["torch", "transformers", "pandas", "numpy"],
        )
        
        if os.path.exists(temp_model_path): shutil.rmtree(temp_model_path)
        run_id = mlflow.active_run().info.run_id

    # Promotion Decision
    client = MlflowClient()
    candidate_version = get_candidate_model_version(MODEL_NAME, run_id)

    try:
        prod_metric = get_production_metric(MODEL_NAME, SELECTION_METRIC)
        improvement = candidate_metric - prod_metric
    except RuntimeError:
        prod_metric, improvement = None, float("inf")

    if improvement >= IMPROVEMENT_THRESHOLD:
        print(f"✅ Promoting model. Improvement: {improvement:.4f}")
        for v in client.get_latest_versions(MODEL_NAME, stages=["Production"]):
            client.transition_model_version_stage(MODEL_NAME, v.version, stage="Archived")
        client.transition_model_version_stage(MODEL_NAME, candidate_version.version, stage="Production")
    else:
        print(f"❌ Keeping current model. Improvement ({improvement:.4f}) below threshold.")

    return history

if __name__ == "__main__":
    run_text_training()
