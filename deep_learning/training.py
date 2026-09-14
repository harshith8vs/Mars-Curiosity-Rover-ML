"""Training engine and unified execution pipeline for deep learning models."""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import config
from deep_learning.evaluation import evaluate_split
from deep_learning.models.resnet50 import (
    freeze_backbone,
    unfreeze_all,
    get_differential_param_groups
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train model for one epoch and return (average_loss, accuracy_percentage)."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size
        
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return float(epoch_loss), float(epoch_acc)


def run_training_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    active_class_names: Dict[int, str],
    warmup_epochs: int = 3,
    finetune_epochs: int = 15,
    warmup_lr: float = 1e-3,
    backbone_lr: float = 5e-5,
    head_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    checkpoint_dir: Optional[config.Path] = None,
    log_callback: Optional[Callable[[str], None]] = None
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, object]]:
    """Execute two-stage fine-tuning (head warmup followed by differential backbone tuning)."""
    if checkpoint_dir is None:
        checkpoint_dir = config.PHASE5A_MODELS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    total_epochs = warmup_epochs + finetune_epochs
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_macro_precision": [],
        "val_macro_recall": [],
        "val_weighted_f1": [],
        "lr": []
    }
    
    best_macro_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1
    best_val_metrics: Dict[str, object] = {}
    best_state_dict = None
    
    log("=" * 86)
    log(f" STARTING RESNET-50 TRAINING: {total_epochs} EPOCHS ({warmup_epochs} WARMUP + {finetune_epochs} FINE-TUNE)")
    log(f" Device: {device} | Batch Size: {train_loader.batch_size} | Head LR: {head_lr} | Backbone LR: {backbone_lr}")
    log("=" * 86)
    log(f"{'Epoch':<8}{'Phase':<12}{'Train Loss':<13}{'Train Acc':<13}{'Val Loss':<12}{'Val Acc':<12}{'Val Macro-F1':<14}{'Status'}")
    log("-" * 86)
    
    start_total_time = time.time()
    
    # Stage 1: Head warmup (backbone frozen)
    if warmup_epochs > 0:
        freeze_backbone(model)
        optimizer_warmup = AdamW(model.fc.parameters(), lr=warmup_lr, weight_decay=weight_decay)
        
        for epoch in range(1, warmup_epochs + 1):
            t_epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_warmup, device)
            val_metrics, _, _, _ = evaluate_split(model, val_loader, criterion, device, active_class_names)
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["val_macro_precision"].append(val_metrics["macro_precision"])
            history["val_macro_recall"].append(val_metrics["macro_recall"])
            history["val_weighted_f1"].append(val_metrics["weighted_f1"])
            history["lr"].append(warmup_lr)
            
            is_best = False
            if (val_metrics["macro_f1"] > best_macro_f1) or (
                np.isclose(val_metrics["macro_f1"], best_macro_f1) and val_metrics["accuracy"] > best_acc
            ):
                best_macro_f1 = val_metrics["macro_f1"]
                best_acc = val_metrics["accuracy"]
                best_epoch = epoch
                best_val_metrics = copy.deepcopy(val_metrics)
                best_state_dict = copy.deepcopy(model.state_dict())
                is_best = True
                
            status_str = f"★ BEST (F1: {best_macro_f1:.4f})" if is_best else ""
            log(f"{epoch:<8}{'Warmup':<12}{train_loss:<13.4f}{train_acc:<12.2f}%{val_metrics['loss']:<12.4f}{val_metrics['accuracy']:<11.2f}%{val_metrics['macro_f1']:<14.4f}{status_str}")
            
    # Stage 2: Backbone fine-tuning (all layers unfrozen)
    if finetune_epochs > 0:
        unfreeze_all(model)
        param_groups = get_differential_param_groups(model, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
        optimizer_ft = AdamW(param_groups)
        scheduler = CosineAnnealingLR(optimizer_ft, T_max=finetune_epochs, eta_min=1e-6)
        
        for stage2_epoch in range(1, finetune_epochs + 1):
            epoch = warmup_epochs + stage2_epoch
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_ft, device)
            val_metrics, _, _, _ = evaluate_split(model, val_loader, criterion, device, active_class_names)
            
            current_lr = optimizer_ft.param_groups[0]["lr"]
            scheduler.step()
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["val_macro_precision"].append(val_metrics["macro_precision"])
            history["val_macro_recall"].append(val_metrics["macro_recall"])
            history["val_weighted_f1"].append(val_metrics["weighted_f1"])
            history["lr"].append(current_lr)
            
            is_best = False
            if (val_metrics["macro_f1"] > best_macro_f1) or (
                np.isclose(val_metrics["macro_f1"], best_macro_f1) and val_metrics["accuracy"] > best_acc
            ):
                best_macro_f1 = val_metrics["macro_f1"]
                best_acc = val_metrics["accuracy"]
                best_epoch = epoch
                best_val_metrics = copy.deepcopy(val_metrics)
                best_state_dict = copy.deepcopy(model.state_dict())
                is_best = True
                
            status_str = f"★ BEST (F1: {best_macro_f1:.4f})" if is_best else ""
            log(f"{epoch:<8}{'FineTune':<12}{train_loss:<13.4f}{train_acc:<12.2f}%{val_metrics['loss']:<12.4f}{val_metrics['accuracy']:<11.2f}%{val_metrics['macro_f1']:<14.4f}{status_str}")
            
    total_time = time.time() - start_total_time
    log("-" * 86)
    log(f" TRAINING COMPLETE in {total_time:.1f}s ({total_time/60.0:.2f} min).")
    log(f" Best Checkpoint: Epoch {best_epoch} | Validation Macro-F1 = {best_macro_f1:.4f} | Validation Accuracy = {best_acc:.2f}%")
    log("=" * 86)
    
    # Save Best and Last Checkpoints
    best_ckpt_path = checkpoint_dir / "best_resnet50.pth"
    last_ckpt_path = checkpoint_dir / "last_resnet50.pth"
    
    torch.save({
        "epoch": best_epoch,
        "state_dict": best_state_dict,
        "val_macro_f1": best_macro_f1,
        "val_accuracy": best_acc,
        "val_metrics": best_val_metrics,
        "history": history
    }, best_ckpt_path)
    log(f" Saved best checkpoint to: {best_ckpt_path}")
    
    torch.save({
        "epoch": total_epochs,
        "state_dict": model.state_dict(),
        "history": history
    }, last_ckpt_path)
    log(f" Saved last checkpoint to: {last_ckpt_path}")
    
    # Load best weights into model before returning
    model.load_state_dict(best_state_dict)
    
    best_val_summary = {
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "training_time_seconds": float(total_time),
        "validation_macro_f1": float(best_macro_f1),
        "validation_accuracy": float(best_acc),
        "validation_loss": float(best_val_metrics["loss"]),
        "validation_macro_precision": float(best_val_metrics["macro_precision"]),
        "validation_macro_recall": float(best_val_metrics["macro_recall"]),
        "validation_weighted_f1": float(best_val_metrics["weighted_f1"]),
        "per_class": best_val_metrics["per_class"],
        "confusion_matrix": best_val_metrics["confusion_matrix"]
    }
    
    return model, history, best_val_summary


class DeepLearningPipeline:
    """Unified execution pipeline for ResNet-50, MRSCAtt, ViT, and EfficientNet-B3."""

    BENCHMARK_CONFIGS = {
        "resnet50": {
            "display_name": "ResNet-50 (Deep Learning Baseline)",
            "checkpoint": config.MODELS_RESNET50_DIR / "best_resnet50.pth",
            "resolution": (224, 224),
            "benchmark_acc": 77.78,
            "benchmark_macro_f1": 0.6631,
            "predictions_dir": config.RESULTS_RESNET50_DIR / "predictions"
        },
        "mrscatt": {
            "display_name": "MRSCAtt (Spatial & Channel Attention)",
            "checkpoint": config.MODELS_MRSCATT_DIR / "best_mrscatt.pth",
            "resolution": (224, 224),
            "benchmark_acc": 64.29,
            "benchmark_macro_f1": 0.5851,
            "predictions_dir": config.RESULTS_MRSCATT_DIR / "predictions"
        },
        "vit": {
            "display_name": "ViT-B/16 (Vision Transformer)",
            "checkpoint": config.MODELS_VIT_DIR / "best_vit.pth",
            "resolution": (224, 224),
            "benchmark_acc": 73.26,
            "benchmark_macro_f1": 0.6536,
            "predictions_dir": config.RESULTS_VIT_DIR / "predictions"
        },
        "efficientnet_b3": {
            "display_name": "EfficientNet-B3 (Champion Model)",
            "checkpoint": config.MODELS_EFFICIENTNET_B3_DIR / "best_efficientnet_b3.pth",
            "resolution": (300, 300),
            "benchmark_acc": 80.61,
            "benchmark_macro_f1": 0.7083,
            "predictions_dir": config.RESULTS_EFFICIENTNET_B3_DIR / "predictions"
        }
    }

    def __init__(self):
        from deep_learning.datasets import (
            load_split_records,
            load_synset_mapping,
            build_active_class_mappings
        )
        self.synset_mapping = load_synset_mapping()
        self._splits = {
            "train": load_split_records(config.TRAIN_LABELS_PATH),
            "val": load_split_records(config.VAL_LABELS_PATH),
            "test": load_split_records(config.TEST_LABELS_PATH)
        }
        self.orig_to_active, self.active_to_orig, self.active_class_names, self.active_classes = build_active_class_mappings(
            self._splits["train"], self.synset_mapping
        )

    def build_model(self, model_key: str, pretrained: bool = False) -> nn.Module:
        """Constructs uninitialized or pretrained model architecture."""
        clean_key = model_key.lower().replace("-", "_").replace(" ", "_")
        if clean_key == "resnet50":
            from deep_learning.models.resnet50 import build_resnet50_baseline
            model, _ = build_resnet50_baseline(num_classes=24, pretrained=pretrained)
        elif clean_key == "mrscatt":
            from deep_learning.models.mrscatt import build_mrscatt_model
            model, _ = build_mrscatt_model(num_classes=24, pretrained=pretrained)
        elif clean_key == "vit":
            from deep_learning.models.vit import build_vit_baseline
            model, _ = build_vit_baseline(num_classes=24, pretrained=pretrained)
        elif clean_key in ["efficientnet_b3", "efficientnet"]:
            from deep_learning.models.efficientnet_b3 import build_efficientnet_b3_baseline
            model, _ = build_efficientnet_b3_baseline(num_classes=24, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown deep learning model: {model_key}. Choose from {list(self.BENCHMARK_CONFIGS.keys())}")
        return model

    def load_checkpoint(self, model_key: str, checkpoint_path: Optional[Path] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Instantiates architecture and loads saved model checkpoint without downloading weights."""
        clean_key = model_key.lower().replace("-", "_").replace(" ", "_")
        if clean_key not in self.BENCHMARK_CONFIGS and clean_key != "efficientnet":
            raise ValueError(f"Unknown model: {model_key}")
        if clean_key == "efficientnet":
            clean_key = "efficientnet_b3"

        if checkpoint_path is None:
            checkpoint_path = self.BENCHMARK_CONFIGS[clean_key]["checkpoint"]

        model = self.build_model(clean_key, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(sd)
        return model, checkpoint

    def get_dataloader(self, model_key: str, split: str = "test", batch_size: int = 32, num_workers: int = 0) -> DataLoader:
        """Constructs DataLoader for specified model and split."""
        clean_key = "efficientnet_b3" if model_key.lower().replace("-", "_") in ["efficientnet", "efficientnet_b3"] else model_key.lower().replace("-", "_")
        cfg = self.BENCHMARK_CONFIGS[clean_key]
        resolution = cfg["resolution"]

        from deep_learning.datasets import MarsRoverDataset
        from deep_learning.transforms import get_eval_transforms

        dataset = MarsRoverDataset(
            records=self._splits[split],
            calibrated_dir=config.CALIBRATED_IMG_DIR,
            original_to_active=self.orig_to_active,
            transform=get_eval_transforms(resolution)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def recompute_from_predictions(self, model_key: str) -> Dict[str, Any]:
        """Mathematically verifies benchmark metrics directly from saved prediction arrays (<0.1s)."""
        clean_key = "efficientnet_b3" if model_key.lower().replace("-", "_") in ["efficientnet", "efficientnet_b3"] else model_key.lower().replace("-", "_")
        cfg = self.BENCHMARK_CONFIGS[clean_key]
        pred_dir = cfg["predictions_dir"]

        y_true = np.load(pred_dir / "test_labels.npy")
        y_pred = np.load(pred_dir / "test_predictions.npy")

        from deep_learning.evaluation import recompute_metrics_from_predictions
        m = recompute_metrics_from_predictions(y_true, y_pred, active_class_names=self.active_class_names)
        m["model_key"] = clean_key
        m["display_name"] = cfg["display_name"]
        m["benchmark_acc"] = cfg["benchmark_acc"]
        m["benchmark_macro_f1"] = cfg["benchmark_macro_f1"]
        return m

    def evaluate_model(
        self,
        model_key: str,
        split: str = "test",
        device: Optional[torch.device] = None,
        generate_artifacts: bool = False
    ) -> Dict[str, Any]:
        """Evaluates saved model checkpoint on live dataset split and optionally regenerates result artifacts."""
        clean_key = "efficientnet_b3" if model_key.lower().replace("-", "_") in ["efficientnet", "efficientnet_b3"] else model_key.lower().replace("-", "_")
        from deep_learning.utils import (
            get_device,
            plot_confusion_matrix,
            plot_normalized_confusion_matrix,
            plot_per_class_f1_barchart,
            save_classification_report
        )
        dev = device or get_device()

        model, _ = self.load_checkpoint(clean_key)
        model.to(dev)
        dataloader = self.get_dataloader(clean_key, split=split)

        from deep_learning.evaluation import evaluate_split
        metrics, all_preds, all_targets, all_probs = evaluate_split(
            model=model,
            dataloader=dataloader,
            criterion=None,
            device=dev,
            active_class_names=self.active_class_names
        )
        cfg = self.BENCHMARK_CONFIGS[clean_key]
        metrics["model_key"] = clean_key
        metrics["display_name"] = cfg["display_name"]
        metrics["benchmark_acc"] = cfg["benchmark_acc"]
        metrics["benchmark_macro_f1"] = cfg["benchmark_macro_f1"]

        if generate_artifacts:
            import json
            dir_map = {
                "resnet50": config.RESULTS_RESNET50_DIR,
                "mrscatt": config.RESULTS_MRSCATT_DIR,
                "vit": config.RESULTS_VIT_DIR,
                "efficientnet_b3": config.RESULTS_EFFICIENTNET_B3_DIR
            }
            out_dir = dir_map[clean_key]
            out_dir.mkdir(parents=True, exist_ok=True)

            cm_dir = out_dir / "confusion_matrices"
            pred_dir = out_dir / "predictions"
            plots_dir = out_dir / "plots"
            reports_dir = out_dir / "classification_reports"
            for d in [cm_dir, pred_dir, plots_dir, reports_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # Prediction arrays
            np.save(pred_dir / f"{split}_predictions.npy", all_preds)
            np.save(pred_dir / f"{split}_labels.npy", all_targets)
            np.save(pred_dir / f"{split}_probabilities.npy", all_probs)

            # Confusion matrices
            cm = np.array(metrics["confusion_matrix"])
            class_labels = [self.active_class_names.get(i, f"Class {i}") for i in range(24)]
            np.save(cm_dir / f"{split}_confusion_matrix.npy", cm)
            plot_confusion_matrix(
                cm=cm,
                class_labels=class_labels,
                title=f"{cfg['display_name']} — Confusion Matrix ({split.capitalize()} Split)",
                save_path=cm_dir / f"{split}_confusion_matrix.png"
            )
            plot_normalized_confusion_matrix(
                cm=cm,
                class_labels=class_labels,
                title=f"{cfg['display_name']} — Normalized Confusion Matrix ({split.capitalize()} Split)",
                save_path=cm_dir / f"{split}_confusion_matrix_normalized.png"
            )

            # Per-class F1 bar chart
            plot_per_class_f1_barchart(
                per_class_dict=metrics["per_class"],
                save_path=plots_dir / f"per_class_f1_{split}_barchart.png",
                title=f"{cfg['display_name']} — Per-Class F1 Scores ({split.capitalize()})"
            )

            # Classification reports (CSV and TXT)
            save_classification_report(
                per_class_dict=metrics["per_class"],
                overall_metrics=metrics,
                csv_path=reports_dir / f"{split}_classification_report.csv",
                txt_path=reports_dir / f"{split}_classification_report.txt"
            )

            # Final metrics JSON
            with open(out_dir / "final_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        return metrics

    def evaluate_all(self, fast: bool = False, generate_artifacts: bool = False) -> List[Dict[str, Any]]:
        """Evaluate all 4 deep learning models (fast recomputes from predictions, live runs forward pass)."""
        results = []
        for key in ["resnet50", "mrscatt", "vit", "efficientnet_b3"]:
            if fast and not generate_artifacts:
                res = self.recompute_from_predictions(key)
            else:
                res = self.evaluate_model(key, split="test", generate_artifacts=generate_artifacts)
            results.append(res)
        return results


# Global singleton instance
dl_pipeline = DeepLearningPipeline()
