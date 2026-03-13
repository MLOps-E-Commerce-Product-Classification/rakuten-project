from __future__ import annotations

import torch.nn as nn
from transformers import AutoModelForSequenceClassification


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
        )
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
        model = AutoModelForSequenceClassification.from_config(config)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model
