from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


SUPPORTED_MODELS = {
    "efficientnet_b0",
    "resnet18",
}


def build_image_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build an image classification model and adapt the classification head.
    """

    model_name = model_name.lower()

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}"
        )

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        if model_name == "efficientnet_b0":
            for param in model.classifier.parameters():
                param.requires_grad = True

        elif model_name == "resnet18":
            for param in model.fc.parameters():
                param.requires_grad = True

    return model


def get_model_transforms(model_name: str, pretrained: bool = True) -> Any | None:
    """
    Return recommended preprocessing transforms for pretrained weights.
    """

    model_name = model_name.lower()

    if not pretrained:
        return None

    if model_name == "efficientnet_b0":
        return EfficientNet_B0_Weights.DEFAULT.transforms()

    if model_name == "resnet18":
        return ResNet18_Weights.DEFAULT.transforms()

    raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")
