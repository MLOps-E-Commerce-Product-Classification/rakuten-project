from __future__ import annotations
from pathlib import Path

import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


SUPPORTED_MODELS = {
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "camembert-base",
}


def build_text_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a text classification model with a classification head.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=Path(model_name).exists(),
        )
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
        model = AutoModelForSequenceClassification.from_config(config)
        model.config.local_files_only = Path(model_name).exists()

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model
