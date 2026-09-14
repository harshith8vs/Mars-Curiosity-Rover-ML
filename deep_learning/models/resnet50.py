"""ResNet-50 baseline model architecture for Mars rover image classification."""

import os
from typing import Dict, List, Tuple
import certifi
import torch
import torch.nn as nn

# Ensure valid SSL certificates on macOS for downloading pretrained checkpoints
if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = certifi.where()

from torchvision.models import resnet50, ResNet50_Weights


def build_resnet50_baseline(
    num_classes: int = 24,
    pretrained: bool = True,
    dropout_rate: float = 0.2
) -> Tuple[nn.Module, Dict[str, object]]:
    """Construct ResNet-50 baseline model adapted to 24 active NASA classes."""
    assert num_classes == 24, f"ResNet-50 head must have 24 active classes, got {num_classes}"
    
    checkpoint_name = "ResNet50_Weights.DEFAULT (IMAGENET1K_V1)" if pretrained else "None (Random Init)"
    
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)
        
    in_features = model.fc.in_features
    assert in_features == 2048, f"Expected 2048 in_features for ResNet-50 fc, got {in_features}"
    
    # Replace final 1000-class classifier with 24-class head
    if dropout_rate > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        final_linear = model.fc[1]
    else:
        model.fc = nn.Linear(in_features, num_classes)
        final_linear = model.fc
        
    # Programmatic assertion of final output dimension
    assert final_linear.out_features == num_classes, (
        f"Classifier output dimension mismatch: {final_linear.out_features} != {num_classes}"
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metadata = {
        "architecture": "ResNet-50",
        "pretrained": pretrained,
        "checkpoint": checkpoint_name,
        "input_channels": 3,
        "input_resolution": (224, 224),
        "in_features": in_features,
        "num_classes": num_classes,
        "dropout_rate": dropout_rate,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }
    
    return model, metadata


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the classification head (model.fc)."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def get_differential_param_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float = 1e-4
) -> List[Dict[str, object]]:
    """Split model parameters into backbone and head groups with differential learning rates."""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay
        })
        
    return param_groups
