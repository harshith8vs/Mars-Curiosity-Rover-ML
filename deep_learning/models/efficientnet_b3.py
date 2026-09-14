"""EfficientNet-B3 model architecture for Mars rover image classification."""

import os
from typing import Dict, Tuple
import certifi

# Ensure valid SSL certificates on macOS for downloading torchvision checkpoints
if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = certifi.where()

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights


def build_efficientnet_b3_baseline(
    num_classes: int = 24,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> Tuple[nn.Module, Dict[str, object]]:
    """Construct EfficientNet-B3 model adapted to 24 active NASA classes."""
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    # Inspect classifier input dimension
    if hasattr(model.classifier, "__getitem__") and hasattr(model.classifier[-1], "in_features"):
        in_features = model.classifier[-1].in_features
    elif hasattr(model.classifier, "in_features"):
        in_features = model.classifier.in_features
    else:
        in_features = 1536

    # Replace classifier with 24-class projection
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    )

    # Verify classifier output features
    assert model.classifier[-1].out_features == num_classes, (
        f"Classifier output features ({model.classifier[-1].out_features}) != num_classes ({num_classes})"
    )

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    metadata: Dict[str, object] = {
        "architecture": "EfficientNet-B3",
        "pretrained": pretrained,
        "checkpoint": str(weights) if pretrained else "None",
        "input_resolution": (300, 300),
        "in_features": in_features,
        "num_classes": num_classes,
        "dropout_rate": dropout_rate,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params
    }

    return model, metadata
