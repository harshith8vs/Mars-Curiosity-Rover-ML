"""Vision Transformer (ViT-B/16) model architecture for Mars rover image classification."""

import os
from typing import Dict, Tuple
import certifi
import torch
import torch.nn as nn

# Ensure valid SSL certificates on macOS for downloading pretrained checkpoints
if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = certifi.where()

from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_vit_baseline(
    num_classes: int = 24,
    pretrained: bool = True,
    dropout_rate: float = 0.0
) -> Tuple[nn.Module, Dict[str, object]]:
    """Construct Vision Transformer (ViT-B/16) model adapted to 24 active NASA classes."""
    assert num_classes == 24, f"ViT head must have 24 active classes, got {num_classes}"
    
    checkpoint_name = "ViT_B_16_Weights.DEFAULT (IMAGENET1K_V1)" if pretrained else "None (Random Init)"
    
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
    else:
        model = vit_b_16(weights=None)
        
    in_features = model.heads.head.in_features
    assert in_features == 768, f"Expected 768 in_features for ViT-B/16 head, got {in_features}"
    
    # Replace final 1000-class classifier with 24-class projection
    if dropout_rate > 0.0:
        model.heads = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        final_linear = model.heads[1]
    else:
        model.heads.head = nn.Linear(in_features, num_classes)
        final_linear = model.heads.head
        
    # Programmatic assertion of final output dimension
    assert final_linear.out_features == num_classes, (
        f"Classifier output dimension mismatch: {final_linear.out_features} != {num_classes}"
    )
    
    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    metadata = {
        "architecture": "Vision Transformer (ViT-B/16)",
        "pretrained": pretrained,
        "checkpoint": checkpoint_name,
        "input_resolution": (224, 224),
        "patch_size": (16, 16),
        "num_patches": 196,
        "hidden_dim": in_features,
        "num_classes": num_classes,
        "dropout_rate": dropout_rate,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params
    }
    
    return model, metadata
