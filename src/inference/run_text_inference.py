from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import torch
import yaml
import numpy as np

from src.data.text_preprocessing import build_tokenizer, preprocess_text
from src.models.text_classifier import build_text_model


RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_mapping(mapping_path: str | Path) -> dict[str, int]:
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if "code_to_idx" in raw:
        return {k: int(v) for k, v in raw["code_to_idx"].items()}
    return {k: (v[0] if isinstance(v, list) else int(v)) for k, v in raw.items()}


def invert_label_mapping(label_encoding: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in label_encoding.items()}


def prepare_text_encoding(
    text_input: str | dict[str, Any],
    tokenizer,
    preprocessing_config_path: str | Path,
) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(text_input, dict):
        designation = text_input.get("designation", "")
        description = text_input.get("description", "")
        processed = preprocess_text(
            designation=designation,
            description=description,
            config_path=preprocessing_config_path,
        )
    else:
        processed = preprocess_text(
            designation=text_input,
            description="",
            config_path=preprocessing_config_path,
        )
    text = processed[0] if isinstance(processed, tuple) else processed

    preprocessing_config = load_config(preprocessing_config_path)
    max_length = int(
        preprocessing_config.get("preprocessing", {}).get("max_length", 128)
    )

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoding, text


def sanitize_float(val: float) -> float:
    """Konvertiert auf float und ersetzt NaN / Inf durch 0.0"""
    val = float(val)
    if not np.isfinite(val):
        return 0.0
    return val


@torch.no_grad()
def predict_single_text(
    model: torch.nn.Module,
    text_input: str | dict[str, Any],
    tokenizer,
    preprocessing_config_path: str | Path,
    device: torch.device,
    idx_to_label: dict[int, str],
    top_k: int = 5,
) -> dict:
    encoding, processed_text = prepare_text_encoding(
        text_input, tokenizer, preprocessing_config_path
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    logits = logits - logits.max(dim=1, keepdim=True).values
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
    probabilities = torch.clamp(probabilities, 0.0, 1.0)

    top_k = min(top_k, probabilities.shape[0])
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    top_k_predictions = [
        {
            "rakuten_code": int(idx_to_label[int(idx)]),  # jetzt als int
            "probability": float(probabilities[idx].item()),
        }
        for idx in top_indices.tolist()
    ]

    predicted_class_idx = int(top_indices[0])
    predicted_label = int(idx_to_label[predicted_class_idx])  # als int

    probabilities_dict = {
        str(int(idx_to_label[i])): float(probabilities[i].item())
        for i in range(len(probabilities))
    }

    return {
        "predicted_rakuten_code": predicted_label,
        "top_k_predictions": top_k_predictions,
        "probabilities": probabilities_dict,
    }


@torch.no_grad()
def predict_multiple_texts(
    model: torch.nn.Module,
    text_inputs: list[str | dict[str, Any]],
    tokenizer,
    preprocessing_config_path: str | Path,
    device: torch.device,
    idx_to_label: dict[int, str],
    top_k: int = 5,
) -> list[dict]:
    return [
        predict_single_text(
            model,
            text_input,
            tokenizer,
            preprocessing_config_path,
            device,
            idx_to_label,
            top_k,
        )
        for text_input in text_inputs
    ]


def save_inference_results(
    results: dict | list[dict],
    output_path: str | Path = RESULTS_PATH / "text_inference_results.json",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_text_inference(
    text_input: str | dict[str, Any] | list[str | dict[str, Any]],
    train_config_path: str | Path = "configs/text_train_config.yaml",
    preprocessing_config_path: str | Path = "configs/text_preprocessing_config.yaml",
    model_weights_path: str | Path = "models/best_text_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
    output_path: str | Path = "results/text_inference_results.json",
    top_k: int = 5,
) -> dict | list[dict]:
    train_config = load_config(train_config_path)
    label_encoding = load_label_mapping(label_encoding_path)
    idx_to_label = invert_label_mapping(label_encoding)
    tokenizer = build_tokenizer(preprocessing_config_path)

    model_config = train_config.get("model", {})
    model_name = model_config.get("name", "bert-base-multilingual-cased")
    num_classes = len(label_encoding)

    model = build_text_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_weights_path = Path(model_weights_path)
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if isinstance(text_input, (str, dict)):
        results = predict_single_text(
            model,
            text_input,
            tokenizer,
            preprocessing_config_path,
            device,
            idx_to_label,
            top_k,
        )
    else:
        results = predict_multiple_texts(
            model,
            text_input,
            tokenizer,
            preprocessing_config_path,
            device,
            idx_to_label,
            top_k,
        )

    save_inference_results(results, output_path)
    return results


if __name__ == "__main__":
    results = run_text_inference(
        text_input=[
            {
                "designation": "T-shirt homme coton manches courtes",
                "description": "T-shirt confortable pour usage quotidien.",
            },
            {
                "designation": "Jeu vidéo action PS4",
                "description": "Edition standard neuve.",
            },
        ],
        train_config_path="configs/text_train_config.yaml",
        preprocessing_config_path="configs/text_preprocessing_config.yaml",
        model_weights_path="models/best_text_model.pt",
        label_encoding_path="configs/label_encoding.json",
        output_path="results/text_inference_results.json",
        top_k=5,
    )

    print("Inference finished.")
    print("Results saved to: results/text_inference_results.json")
