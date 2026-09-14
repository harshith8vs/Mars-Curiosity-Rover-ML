"""Evaluation utilities for computing classification metrics across PyTorch splits."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    active_class_names: Optional[Dict[int, str]] = None
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset split and return metrics, predictions, targets, and probabilities."""
    model.eval()
    
    running_loss = 0.0
    total_samples = 0
    all_preds_list = []
    all_targets_list = []
    all_probs_list = []
    
    t0 = time.time()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)
            
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item() * batch_size
                
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds_list.append(preds.cpu().numpy())
            all_targets_list.append(targets.cpu().numpy())
            all_probs_list.append(probs.cpu().numpy())
            total_samples += batch_size
            
    elapsed_time = time.time() - t0
    
    all_preds = np.concatenate(all_preds_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)
    all_probs = np.concatenate(all_probs_list, axis=0)
    
    assert len(all_preds) == len(dataloader.dataset), (
        f"Evaluation count mismatch: {len(all_preds)} != {len(dataloader.dataset)}"
    )
    
    avg_loss = (running_loss / total_samples) if (criterion is not None and total_samples > 0) else 0.0
    accuracy = float(np.mean(all_preds == all_targets) * 100.0)
    
    # 24 active classes
    num_classes = 24
    labels = list(range(num_classes))
    
    # Aggregate metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    per_p, per_r, per_f1, per_supp = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, average=None, zero_division=0
    )
    
    # Confusion matrix (24 x 24)
    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    
    per_class_dict = {}
    for c_idx in range(num_classes):
        c_name = active_class_names.get(c_idx, f"class_{c_idx}") if active_class_names else f"class_{c_idx}"
        per_class_dict[c_idx] = {
            "active_index": c_idx,
            "class_name": c_name,
            "precision": float(per_p[c_idx]),
            "recall": float(per_r[c_idx]),
            "f1_score": float(per_f1[c_idx]),
            "support": int(per_supp[c_idx])
        }
        
    metrics = {
        "num_samples": total_samples,
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "elapsed_time_seconds": float(elapsed_time),
        "inference_ms_per_sample": float((elapsed_time / total_samples) * 1000.0) if total_samples > 0 else 0.0,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics, all_preds, all_targets, all_probs


def recompute_metrics_from_predictions(
    targets: np.ndarray,
    preds: np.ndarray,
    active_class_names: Optional[Dict[int, str]] = None
) -> Dict[str, object]:
    """Recompute all classification metrics strictly from saved target and prediction arrays."""
    num_classes = 24
    labels = list(range(num_classes))
    
    accuracy = float(np.mean(preds == targets) * 100.0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        targets, preds, labels=labels, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, per_supp = precision_recall_fscore_support(
        targets, preds, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=labels)
    
    per_class_dict = {}
    for c_idx in range(num_classes):
        c_name = active_class_names.get(c_idx, f"class_{c_idx}") if active_class_names else f"class_{c_idx}"
        per_class_dict[c_idx] = {
            "active_index": c_idx,
            "class_name": c_name,
            "precision": float(per_p[c_idx]),
            "recall": float(per_r[c_idx]),
            "f1_score": float(per_f1[c_idx]),
            "support": int(per_supp[c_idx])
        }
        
    return {
        "num_samples": len(targets),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }
