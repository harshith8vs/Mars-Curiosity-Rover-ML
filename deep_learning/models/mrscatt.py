"""MRSCAtt (Mars Rover Spatial and Channel Attention) model architecture."""

import os
from typing import Dict, List, Tuple
import certifi
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure valid SSL certificates on macOS for downloading pretrained checkpoints
if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = certifi.where()

from torchvision.models import resnet50, ResNet50_Weights


class ChannelAttention(nn.Module):
    """Channel attention with dual pooling branches (average and max) and sigmoid scaling."""
    def __init__(self, in_channels: int = 2048, reduction_ratio: int = 16, dropout_rate: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        hidden_dim = max(1, in_channels // reduction_ratio)
        
        # Average pooling branch
        self.avg_fc1 = nn.Linear(in_channels, hidden_dim, bias=False)
        self.avg_bn1 = nn.BatchNorm1d(hidden_dim)
        self.avg_relu = nn.ReLU(inplace=True)
        self.avg_drop1 = nn.Dropout(p=dropout_rate)
        self.avg_fc2 = nn.Linear(hidden_dim, in_channels, bias=False)
        self.avg_bn2 = nn.BatchNorm1d(in_channels)
        self.avg_drop2 = nn.Dropout(p=dropout_rate)
        
        # Max pooling branch (separate weights as specified)
        self.max_fc1 = nn.Linear(in_channels, hidden_dim, bias=False)
        self.max_bn1 = nn.BatchNorm1d(hidden_dim)
        self.max_relu = nn.ReLU(inplace=True)
        self.max_drop1 = nn.Dropout(p=dropout_rate)
        self.max_fc2 = nn.Linear(hidden_dim, in_channels, bias=False)
        self.max_bn2 = nn.BatchNorm1d(in_channels)
        self.max_drop2 = nn.Dropout(p=dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Global pooling across spatial dimensions (H, W)
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(b, c)
        
        # Forward through separate branches
        avg_out = self.avg_drop2(self.avg_bn2(self.avg_fc2(self.avg_drop1(self.avg_relu(self.avg_bn1(self.avg_fc1(avg_pool)))))))
        max_out = self.max_drop2(self.max_bn2(self.max_fc2(self.max_drop1(self.max_relu(self.max_bn1(self.max_fc1(max_pool)))))))
        
        # Sum and Sigmoid activation
        channel_scale = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * channel_scale


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module using Channel Average and Max pooling followed by 7x7 Conv.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, f"Kernel size must be odd, got {kernel_size}"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension (shape: B, 2, H, W)
        cat_out = torch.cat([avg_out, max_out], dim=1)
        
        # 7x7 Conv -> BatchNorm -> Sigmoid
        spatial_scale = torch.sigmoid(self.bn(self.conv(cat_out)))
        return x * spatial_scale


class ChannelSpatialBlock(nn.Module):
    """Sequential channel and spatial attention block (CBAM-style)."""
    def __init__(self, in_channels: int = 2048, reduction_ratio: int = 16, kernel_size: int = 7, dropout_rate: float = 0.2):
        super().__init__()
        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
            dropout_rate=dropout_rate
        )
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channel = self.channel_attention(x)
        x_spatial = self.spatial_attention(x_channel)
        return x_spatial


class MRSCAttNet(nn.Module):
    """MRSCAtt network: ResNet-50 backbone with post-layer4 CBAM attention."""
    def __init__(self, num_classes: int = 24, pretrained: bool = True, dropout_rate: float = 0.2):
        super().__init__()
        assert num_classes == 24, f"Target classes must be 24, got {num_classes}"
        
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_model = resnet50(weights=weights)
        
        # ResNet-50 Backbone Layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # MRSCAtt Attention Block placed after layer4
        self.cbam = ChannelSpatialBlock(
            in_channels=2048,
            reduction_ratio=16,
            kernel_size=7,
            dropout_rate=dropout_rate
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial stages
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Attention block
        x = self.cbam(x)
        
        # Pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def freeze_mrscatt_backbone(model: MRSCAttNet) -> None:
    """Freeze conv1 through layer3, keeping layer4, cbam, and fc trainable."""
    # Freeze stem and stages 1-3
    for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]:
        for param in m.parameters():
            param.requires_grad = False
            
    # Keep layer4, attention block, and head trainable
    for m in [model.layer4, model.cbam, model.fc]:
        for param in m.parameters():
            param.requires_grad = True


def build_mrscatt_model(
    num_classes: int = 24,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    apply_freezing: bool = True
) -> Tuple[MRSCAttNet, Dict[str, object]]:
    """Construct MRSCAtt model, optionally apply backbone freezing, and return metadata."""
    model = MRSCAttNet(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    
    if apply_freezing:
        freeze_mrscatt_backbone(model)
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    metadata = {
        "architecture": "MRSCAtt",
        "backbone": "ResNet-50",
        "pretrained": pretrained,
        "checkpoint": "ResNet50_Weights.DEFAULT (IMAGENET1K_V1)" if pretrained else "None",
        "attention_block": "ChannelSpatialBlock (Separate avg/max branches + 7x7 spatial conv)",
        "input_resolution": (224, 224),
        "num_classes": num_classes,
        "dropout_rate": dropout_rate,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "freezing_configuration": "conv1, bn1, layer1..3 frozen; layer4, cbam, fc trainable"
    }
    
    return model, metadata
