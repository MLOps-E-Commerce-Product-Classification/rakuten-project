from __future__ import annotations

import json
from pathlib import Path

import bentoml
import torch

from src.data.text_preprocessing import build_tokenizer
from src.inference.run_text_inference import invert_label_mapping, load_label_mapping
from src.models.text_classifier import build_text_model
from src.serving.prepare_bento_assets import ensure_local_text_backbone


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "rakuten_text_classifier"
DEFAULT_TRAIN_CONFIG_PATH = BASE_DIR / "configs/text_train_config.yaml"
DEFAULT_PREPROCESSING_CONFIG_PATH = BASE_DIR / "configs/text_preprocessing_config.yaml"
DEFAULT_MODEL_WEIGHTS_PATH = BASE_DIR / "models/best_text_model.pt"
DEFAULT_LABEL_ENCODING_PATH = BASE_DIR / "configs/label_encoding.json"


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def register_text_model(
    model_name: str = DEFAULT_MODEL_NAME,
    train_config_path: str | Path = DEFAULT_TRAIN_CONFIG_PATH,
    preprocessing_config_path: str | Path = DEFAULT_PREPROCESSING_CONFIG_PATH,
    model_weights_path: str | Path = DEFAULT_MODEL_WEIGHTS_PATH,
    label_encoding_path: str | Path = DEFAULT_LABEL_ENCODING_PATH,
) -> bentoml.Model:
    train_config_path = Path(train_config_path)
    preprocessing_config_path = Path(preprocessing_config_path)
    model_weights_path = Path(model_weights_path)
    label_encoding_path = Path(label_encoding_path)

    if not model_weights_path.exists():
        raise FileNotFoundError(
            "Model weights not found. Pull or create models/best_text_model.pt before registering it in BentoML."
        )

    backbone_path = ensure_local_text_backbone()
    train_config = (
        _load_json(train_config_path.with_suffix(".json"))
        if train_config_path.suffix == ".json"
        else None
    )
    if train_config is None:
        import yaml

        with train_config_path.open("r", encoding="utf-8") as file:
            train_config = yaml.safe_load(file)

    label_encoding = load_label_mapping(label_encoding_path)
    idx_to_label = invert_label_mapping(label_encoding)
    tokenizer = build_tokenizer(
        preprocessing_config_path, local_model_dir=backbone_path
    )

    model_config = train_config.get("model", {})
    configured_name = model_config.get("name", "bert-base-multilingual-cased")
    model_name_or_path = (
        str(backbone_path) if backbone_path.exists() else configured_name
    )

    model = build_text_model(
        model_name=model_name_or_path,
        num_classes=len(label_encoding),
        pretrained=False,
        freeze_backbone=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    bento_model = bentoml.pytorch.save_model(
        model_name,
        model,
        labels={"team": "rakuten", "task": "text-classification", "stage": "serving"},
        metadata={
            "weights_path": str(model_weights_path),
            "preprocessing_config_path": str(preprocessing_config_path),
            "label_encoding_path": str(label_encoding_path),
            "num_classes": len(label_encoding),
        },
        custom_objects={
            "tokenizer": tokenizer,
            "idx_to_label": idx_to_label,
        },
    )
    return bento_model


if __name__ == "__main__":
    model_ref = register_text_model()
    print(f"Model saved as: {model_ref}")
