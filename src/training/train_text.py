from pathlib import Path
import logging

import mlflow
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
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

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
        labels = (
            list(range(num_classes))
            if num_classes
            else sorted(set(y_true) | set(y_pred))
        )
        per_class_f1 = f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        metrics["per_class_f1"] = {
            int(label): float(s) for label, s in zip(labels, per_class_f1)
        }
    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int | None = None,
    log_interval: int = 50,  # alle 50 Batches loggen
) -> dict:
    model.train()
    running_loss, total = 0.0, 0
    all_labels, all_preds = [], []
    num_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        preds = torch.argmax(outputs.logits, dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

        if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
            pct = (i + 1) / num_batches * 100
            TRAIN_LOGGER.info(
                f"  Batch {i + 1}/{num_batches} ({pct:.1f}%) | loss={loss.item():.4f}"
            )

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
    log_interval: int = 50,
) -> dict:
    model.eval()
    running_loss, total = 0.0, 0
    all_labels, all_preds = [], []
    num_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        running_loss += outputs.loss.item() * labels.size(0)
        total += labels.size(0)
        preds = torch.argmax(outputs.logits, dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

        if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
            pct = (i + 1) / num_batches * 100
            TRAIN_LOGGER.info(f"  Val Batch {i + 1}/{num_batches} ({pct:.1f}%)")

    metrics = compute_classification_metrics(
        all_labels, all_preds, num_classes, compute_per_class_f1
    )
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

    lr = float(training_config.get("learning_rate", 2e-5))
    epochs = int(training_config.get("num_epochs", 5))
    wd = float(training_config.get("weight_decay", 0.01))
    patience = int(training_config.get("early_stopping_patience", 3))
    compute_per_f1 = metrics_config.get("compute_per_class_f1", False)
    main_metric = metrics_config.get("main_metric", "macro_f1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }
    if compute_per_f1:
        history["val_per_class_f1"] = []

    best_score = float("inf") if main_metric == "loss" else -1.0
    epochs_no_improve = 0
    model_save_path = Path(model_save_path)

    TRAIN_LOGGER.info(f"Text training started on {device}")
    TRAIN_LOGGER.info(f"Early stopping patience: {patience}")

    mlflow.log_param("device", str(device))
    mlflow.log_param("early_stopping_patience", patience)

    for epoch in range(epochs):
        TRAIN_LOGGER.info(f"Starting epoch {epoch + 1}/{epochs}")

        train_m = train_one_epoch(
            model, train_dataloader, optimizer, device, num_classes
        )
        val_m = validate_one_epoch(
            model, val_dataloader, device, num_classes, compute_per_f1
        )

        for k in ["loss", "accuracy", "macro_f1"]:
            history[f"train_{k}"].append(train_m[k])
            history[f"val_{k}"].append(val_m[k])
        if compute_per_f1:
            history["val_per_class_f1"].append(val_m["per_class_f1"])

        TRAIN_LOGGER.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_m['loss']:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"val_f1={val_m['macro_f1']:.4f}"
        )

        mlflow.log_metrics(
            {
                "train_loss": train_m["loss"],
                "train_accuracy": train_m["accuracy"],
                "train_macro_f1": train_m["macro_f1"],
                "val_loss": val_m["loss"],
                "val_accuracy": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
            },
            step=epoch,
        )

        curr_score = val_m[main_metric]
        is_better = (
            curr_score < best_score
            if main_metric == "loss"
            else curr_score > best_score
        )

        if is_better:
            best_score = curr_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            TRAIN_LOGGER.info(f"Saved best model with {main_metric}={best_score:.4f}")
            mlflow.log_metric("best_" + main_metric, best_score)
        else:
            epochs_no_improve += 1
            TRAIN_LOGGER.info(
                f"No improvement for {epochs_no_improve}/{patience} epochs"
            )
            if epochs_no_improve >= patience:
                TRAIN_LOGGER.info(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best {main_metric}={best_score:.4f}"
                )
                mlflow.log_param("early_stopped_at_epoch", epoch + 1)
                break

    TRAIN_LOGGER.info("Text training finished")
    return model, history
