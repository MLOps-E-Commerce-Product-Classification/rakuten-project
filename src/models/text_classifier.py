from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


SUPPORTED_MODELS = {
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "camembert-base",
}


def _validate_model_name(model_name: str) -> None:
    if model_name in SUPPORTED_MODELS:
        return
    if Path(model_name).exists():
        return
    raise ValueError(
        f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS} or a local Hugging Face model directory"
    )


def build_text_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a text classification model with a classification head.
    """
    _validate_model_name(model_name)

    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=Path(model_name).exists(),
        )
    else:
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_classes,
            local_files_only=Path(model_name).exists(),
        )
        model = AutoModelForSequenceClassification.from_config(config)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model