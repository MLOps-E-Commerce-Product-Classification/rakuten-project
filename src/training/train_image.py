from pathlib import Path
import logging

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


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

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


TRAIN_LOGGER = setup_logger("training", LOG_PATH / "training.log")


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
        if num_classes is None:
            labels = sorted(set(y_true) | set(y_pred))
        else:
            labels = list(range(num_classes))

        per_class_f1 = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        metrics["per_class_f1"] = {
            int(label): float(score) for label, score in zip(labels, per_class_f1)
        }

    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int | None = None,
) -> dict:
    model.train()

    running_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc="Training Batches", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if total == 0:
        raise ValueError("Dataloader is empty. Cannot compute training metrics.")

    epoch_loss = running_loss / total
    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        num_classes=num_classes,
        compute_per_class_f1=False,
    )
    metrics["loss"] = float(epoch_loss)

    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int | None = None,
    compute_per_class_f1: bool = False,
) -> dict:
    model.eval()

    running_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc="Validation Batches", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    if total == 0:
        raise ValueError("Dataloader is empty. Cannot compute validation metrics.")

    epoch_loss = running_loss / total
    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        num_classes=num_classes,
        compute_per_class_f1=compute_per_class_f1,
    )
    metrics["loss"] = float(epoch_loss)

    return metrics


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config_path: str | Path,
    num_classes: int,
    model_save_path: str | Path = MODEL_PATH / "best_model.pt",
) -> tuple[nn.Module, dict]:
    config = load_train_config(config_path)

    training_config = config.get("training", {})
    metrics_config = config.get("metrics", {})

    learning_rate = training_config.get("learning_rate", 1e-3)
    num_epochs = training_config.get("num_epochs", 10)
    weight_decay = training_config.get("weight_decay", 0.0)

    compute_per_class_f1 = metrics_config.get("compute_per_class_f1", False)
    main_metric = metrics_config.get("main_metric", "macro_f1")

    if main_metric not in {"macro_f1", "accuracy", "loss"}:
        raise ValueError(
            f"Unsupported main_metric '{main_metric}'. "
            "Choose from {'macro_f1', 'accuracy', 'loss'}."
        )

    TRAIN_LOGGER.info(
        f"Config: learning_rate={learning_rate}, "
        f"num_epochs={num_epochs}, "
        f"weight_decay={weight_decay}, "
        f"main_metric={main_metric}, "
        f"compute_per_class_f1={compute_per_class_f1}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    if compute_per_class_f1:
        history["val_per_class_f1"] = []

    if main_metric == "loss":
        best_val_score = float("inf")
    else:
        best_val_score = -1.0

    model_save_path = Path(model_save_path)

    TRAIN_LOGGER.info(f"Training started on device={device}")

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            compute_per_class_f1=compute_per_class_f1,
        )

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
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        current_val_score = val_metrics[main_metric]

        if main_metric == "loss":
            is_better = current_val_score < best_val_score
        else:
            is_better = current_val_score > best_val_score

        if is_better:
            best_val_score = current_val_score
            torch.save(model.state_dict(), model_save_path)

            TRAIN_LOGGER.info(
                f"New best model saved to {model_save_path} "
                f"with val_{main_metric}={best_val_score:.4f}"
            )

    TRAIN_LOGGER.info("Training finished")

    return model, history
