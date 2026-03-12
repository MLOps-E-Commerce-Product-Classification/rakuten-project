from pathlib import Path
import json
import logging
import time

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader


LOG_PATH = Path("logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


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


EVAL_LOGGER = setup_logger("evaluation", LOG_PATH / "evaluation.log")


def load_evaluation_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int | None = None,
    compute_per_class_f1: bool = True,
    compute_confusion_matrix_flag: bool = True,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if num_classes is None:
        labels = sorted(set(y_true) | set(y_pred))
    else:
        labels = list(range(num_classes))

    if compute_per_class_f1:
        per_class_f1 = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        metrics["per_class_f1"] = {
            int(label): float(score)
            for label, score in zip(labels, per_class_f1)
        }

    if compute_confusion_matrix_flag:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics["confusion_matrix"] = cm.tolist()

    return metrics


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_inference_performance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup_batches: int = 2,
    num_measure_batches: int = 10,
) -> dict:
    """
    Measure inference latency and throughput on a subset of batches.
    """
    model.eval()
    measured_batch_times = []
    measured_num_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)

        # warmup
        if batch_idx < num_warmup_batches:
            _synchronize_device(device)
            _ = model(images)
            _synchronize_device(device)
            continue

        if len(measured_batch_times) >= num_measure_batches:
            break

        _synchronize_device(device)
        start_time = time.perf_counter()

        _ = model(images)

        _synchronize_device(device)
        end_time = time.perf_counter()

        batch_time = end_time - start_time
        measured_batch_times.append(batch_time)
        measured_num_samples += images.size(0)

    if not measured_batch_times or measured_num_samples == 0:
        raise ValueError(
            "Not enough batches in dataloader to measure inference performance."
        )

    total_time = sum(measured_batch_times)
    avg_batch_latency_ms = (total_time / len(measured_batch_times)) * 1000.0
    avg_sample_latency_ms = (total_time / measured_num_samples) * 1000.0
    throughput_samples_per_sec = measured_num_samples / total_time

    return {
        "avg_batch_latency_ms": float(avg_batch_latency_ms),
        "avg_sample_latency_ms": float(avg_sample_latency_ms),
        "throughput_samples_per_sec": float(throughput_samples_per_sec),
        "num_measured_batches": int(len(measured_batch_times)),
        "num_measured_samples": int(measured_num_samples),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config_path: str | Path,
    num_classes: int | None = None,
) -> dict:
    """
    Evaluate a trained model on a dataset.
    """
    config = load_evaluation_config(config_path)

    metrics_config = config.get("metrics", {})
    performance_config = config.get("performance", {})

    compute_per_class_f1 = metrics_config.get("compute_per_class_f1", True)
    compute_confusion_matrix_flag = metrics_config.get("compute_confusion_matrix", True)
    main_metric = metrics_config.get("main_metric", "macro_f1")

    measure_latency = performance_config.get("measure_latency", True)
    measure_throughput = performance_config.get("measure_throughput", True)
    latency_num_warmup_batches = performance_config.get(
        "latency_num_warmup_batches", 2
    )
    latency_num_measure_batches = performance_config.get(
        "latency_num_measure_batches", 10
    )

    if main_metric not in {"macro_f1", "accuracy", "loss"}:
        raise ValueError(
            f"Unsupported main_metric '{main_metric}'. "
            "Choose from {'macro_f1', 'accuracy', 'loss'}."
        )

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []
    all_image_ids = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

        if "image_id" in batch:
            all_image_ids.extend(batch["image_id"])

    if total == 0:
        raise ValueError("Dataloader is empty. Cannot compute evaluation metrics.")

    eval_loss = running_loss / total

    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        num_classes=num_classes,
        compute_per_class_f1=compute_per_class_f1,
        compute_confusion_matrix_flag=compute_confusion_matrix_flag,
    )
    metrics["loss"] = float(eval_loss)

    performance = {}
    if measure_latency or measure_throughput:
        performance = measure_inference_performance(
            model=model,
            dataloader=dataloader,
            device=device,
            num_warmup_batches=latency_num_warmup_batches,
            num_measure_batches=latency_num_measure_batches,
        )

    results = {
        "main_metric": main_metric,
        "main_metric_value": float(metrics[main_metric]),
        "metrics": metrics,
        "performance": performance,
        "predictions": {
            "image_ids": all_image_ids,
            "y_true": all_labels,
            "y_pred": all_preds,
        },
    }

    log_message = (
        f"Evaluation finished | "
        f"loss={metrics['loss']:.4f} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )

    if performance:
        log_message += (
            f" | avg_sample_latency_ms={performance['avg_sample_latency_ms']:.4f}"
            f" | throughput_samples_per_sec={performance['throughput_samples_per_sec']:.4f}"
        )

    EVAL_LOGGER.info(log_message)

    return results


def save_evaluation_results(
    results: dict,
    output_path: str | Path = RESULTS_PATH / "evaluation_results.json",
) -> None:
    """
    Save evaluation results as JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    EVAL_LOGGER.info(f"Evaluation results saved to {output_path}")