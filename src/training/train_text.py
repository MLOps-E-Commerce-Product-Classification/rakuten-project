from pathlib import Path
import logging

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


LOG_PATH = Path("logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: str | Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


TRAIN_LOGGER = setup_logger("text_training", LOG_PATH / "text_training.log")


def load_train_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Train config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int | None = None,
    compute_per_class_f1: bool = False,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if compute_per_class_f1:
        labels = list(range(num_classes)) if num_classes else sorted(set(y_true) | set(y_pred))
        per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        metrics["per_class_f1"] = {int(l): float(s) for l, s in zip(labels, per_class_f1)}

    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int | None = None,
) -> dict:
    model.train()

    running_loss = 0.0
    total = 0
    all_labels, all_preds = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    if total == 0:
        raise ValueError("Dataloader is empty.")

    metrics = compute_classification_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = running_loss / total
    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int | None = None,
    compute_per_class_f1: bool = False,
) -> dict:
    model.eval()

    running_loss = 0.0
    total = 0
    all_labels, all_preds = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    if total == 0:
        raise ValueError("Dataloader is empty.")

    metrics = compute_classification_metrics(all_labels, all_preds, num_classes, compute_per_class_f1)
    metrics["loss"] = running_loss / total
    return metrics


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config_path: str | Path,
    num_classes: int,
    model_save_path: str | Path = MODEL_PATH / "best_text_model.pt",
) -> tuple[nn.Module, dict]:
    config = load_train_config(config_path)

    training_config = config.get("training", {})
    metrics_config = config.get("metrics", {})

    learning_rate = float(training_config.get("learning_rate", 2e-5))
    num_epochs = int(training_config.get("num_epochs", 5))
    weight_decay = float(training_config.get("weight_decay", 0.01))
    compute_per_class_f1 = metrics_config.get("compute_per_class_f1", False)
    main_metric = metrics_config.get("main_metric", "macro_f1")

    if main_metric not in {"macro_f1", "accuracy", "loss"}:
        raise ValueError(f"Unsupported main_metric '{main_metric}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [], "train_accuracy": [], "train_macro_f1": [],
        "val_loss": [], "val_accuracy": [], "val_macro_f1": [],
    }

    if compute_per_class_f1:
        history["val_per_class_f1"] = []

    best_val_score = float("inf") if main_metric == "loss" else -1.0
    model_save_path = Path(model_save_path)

    TRAIN_LOGGER.info(f"Text training started on device={device}")

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_dataloader, optimizer, device, num_classes)
        val_metrics = validate_one_epoch(model, val_dataloader, device, num_classes, compute_per_class_f1)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        if compute_per_class_f1:
            history["val_per_class_f1"].append(val_metrics["per_class_f1"])

        TRAIN_LOGGER.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        current_val_score = val_metrics[main_metric]
        is_better = current_val_score < best_val_score if main_metric == "loss" else current_val_score > best_val_score

        if is_better:
            best_val_score = current_val_score
            torch.save(model.state_dict(), model_save_path)
            TRAIN_LOGGER.info(f"New best model saved to {model_save_path} with val_{main_metric}={best_val_score:.4f}")

    TRAIN_LOGGER.info("Text training finished")
    return model, history
