"""
Training engine and unified execution pipeline for deep learning models.

Architecture-aware helpers handle classifier head access and backbone freezing
correctly for each of the four supported architectures:
  - ResNet-50:        classifier = model.fc
  - MRSCAtt:          classifier = model.fc  (custom attention + ResNet backbone)
  - ViT-B/16:         classifier = model.heads.head
  - EfficientNet-B3:  classifier = model.classifier  (Sequential, last layer is Linear)

BatchNorm behavior during frozen backbone:
  When the backbone is frozen (requires_grad=False), BatchNorm layers in the
  frozen portion are also set to eval() mode to prevent running statistics
  (mean/var) from updating during model.train(). Trainable layers keep their
  BatchNorm in train() mode normally.
"""

import copy
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


# ──────────────────────────────────────────────────────────────────────────────
# Architecture-aware helper functions
# ──────────────────────────────────────────────────────────────────────────────

_SUPPORTED_MODEL_TYPES = {"resnet50", "mrscatt", "vit", "efficientnet_b3"}


def _validate_model_type(model_type: str) -> str:
    """Normalize and validate model type string."""
    clean = model_type.lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    if clean not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Choose from {sorted(_SUPPORTED_MODEL_TYPES)}"
        )
    return clean


def get_classifier_module(model: nn.Module, model_type: str) -> nn.Module:
    """
    Return the classification head module for the given architecture.

    ResNet-50:        model.fc  (Sequential[Dropout, Linear] or Linear)
    MRSCAtt:          model.fc  (Linear)
    ViT-B/16:         model.heads.head  (Linear)
    EfficientNet-B3:  model.classifier  (Sequential[Dropout, Linear])
    """
    mt = _validate_model_type(model_type)
    if mt in ("resnet50", "mrscatt"):
        return model.fc
    elif mt == "vit":
        return model.heads.head
    elif mt == "efficientnet_b3":
        return model.classifier
    raise ValueError(f"Unreachable: {mt}")


def get_classifier_parameters(model: nn.Module, model_type: str) -> List[nn.Parameter]:
    """Return the list of trainable parameters belonging to the classifier head."""
    clf = get_classifier_module(model, model_type)
    return list(clf.parameters())


def freeze_backbone(model: nn.Module, model_type: str) -> None:
    """
    Freeze backbone parameters (requires_grad=False) and set all BatchNorm
    layers in the frozen backbone to eval() mode to prevent running stats update.

    The classifier head is kept trainable (requires_grad=True).

    Architecture-specific frozen/trainable split:
      ResNet-50:        freeze everything except model.fc
      MRSCAtt:          freeze conv1, bn1, layer1-3; keep layer4, cbam, fc trainable
      ViT-B/16:         freeze everything except model.heads
      EfficientNet-B3:  freeze everything except model.classifier
    """
    mt = _validate_model_type(model_type)

    if mt == "resnet50":
        for name, param in model.named_parameters():
            param.requires_grad = "fc" in name

    elif mt == "mrscatt":
        # Freeze stem and stages 1-3
        frozen_modules = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]
        for mod in frozen_modules:
            for param in mod.parameters():
                param.requires_grad = False
        # Keep layer4, cbam, fc trainable
        for mod in [model.layer4, model.cbam, model.fc]:
            for param in mod.parameters():
                param.requires_grad = True

    elif mt == "vit":
        for name, param in model.named_parameters():
            param.requires_grad = "heads" in name

    elif mt == "efficientnet_b3":
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name

    # ── BatchNorm stabilization in frozen backbone ──────────────────────────
    # Any BatchNorm/BatchNorm2d layer whose parameters are all frozen is set
    # to eval() mode so running mean/var are not updated during model.train().
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            all_frozen = all(not p.requires_grad for p in module.parameters())
            if all_frozen:
                module.eval()


def unfreeze_backbone(model: nn.Module, model_type: str) -> None:
    """
    Unfreeze all parameters and restore all BatchNorm layers to train() mode.
    Call before Stage 2 (differential fine-tuning).
    """
    _validate_model_type(model_type)
    for param in model.parameters():
        param.requires_grad = True
    # Restore all BN layers to training mode
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()


def get_parameter_groups(
    model: nn.Module,
    model_type: str,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float = 1e-4,
) -> List[Dict[str, Any]]:
    """
    Build differential learning-rate parameter groups:
      - Backbone (frozen params excluded): backbone_lr
      - Classifier head:                  head_lr

    Architecture-specific "head" name pattern:
      ResNet-50 / MRSCAtt:  "fc"
      ViT-B/16:             "heads"
      EfficientNet-B3:      "classifier"
    """
    mt = _validate_model_type(model_type)
    head_key = {
        "resnet50": "fc",
        "mrscatt": "fc",
        "vit": "heads",
        "efficientnet_b3": "classifier",
    }[mt]

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if head_key in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups


# ──────────────────────────────────────────────────────────────────────────────
# One-epoch training pass
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch; return (average_loss, accuracy_percentage)."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Two-stage training pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(
    model: nn.Module,
    model_type: str,
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
    checkpoint_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, object]]:
    """
    Execute two-stage fine-tuning:
      Stage 1 (Warmup):    Backbone frozen, only classifier head trained.
      Stage 2 (Fine-Tune): All layers unfrozen, differential learning rates applied.

    Model selection criterion: best validation Macro-F1.
    Checkpoint files are saved as best_{model_type}.pth / last_{model_type}.pth.

    NOTE: The repository intentionally excludes trained model checkpoint files
    because of their large size. This pipeline generates checkpoints only in the
    local training environment. Result/prediction artifacts are retained for
    evaluation verification without requiring checkpoints.
    """
    mt = _validate_model_type(model_type)

    if checkpoint_dir is None:
        dir_map = {
            "resnet50": config.MODELS_RESNET50_DIR,
            "mrscatt": config.MODELS_MRSCATT_DIR,
            "vit": config.MODELS_VIT_DIR,
            "efficientnet_b3": config.MODELS_EFFICIENTNET_B3_DIR,
        }
        checkpoint_dir = dir_map[mt]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    total_epochs = warmup_epochs + finetune_epochs
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_macro_f1": [], "val_macro_precision": [],
        "val_macro_recall": [], "val_weighted_f1": [],
        "lr": [],
    }

    best_macro_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1
    best_val_metrics: Dict[str, object] = {}
    best_state_dict = None

    display_name = mt.upper().replace("_", "-")

    log("=" * 86)
    log(f" STARTING {display_name} TRAINING: {total_epochs} EPOCHS "
        f"({warmup_epochs} WARMUP + {finetune_epochs} FINE-TUNE)")
    log(f" Device: {device} | Batch Size: {train_loader.batch_size} "
        f"| Head LR: {head_lr} | Backbone LR: {backbone_lr}")
    log("=" * 86)
    log(f"{'Epoch':<8}{'Phase':<12}{'Train Loss':<13}{'Train Acc':<13}"
        f"{'Val Loss':<12}{'Val Acc':<12}{'Val Macro-F1':<14}{'Status'}")
    log("-" * 86)

    start_total_time = time.time()

    # Stage 1: Head warmup (backbone frozen)
    if warmup_epochs > 0:
        freeze_backbone(model, mt)
        head_params = get_classifier_parameters(model, mt)
        optimizer_warmup = AdamW(head_params, lr=warmup_lr, weight_decay=weight_decay)

        for epoch in range(1, warmup_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer_warmup, device
            )
            val_metrics, _, _, _ = evaluate_split(
                model, val_loader, criterion, device, active_class_names
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["val_macro_precision"].append(val_metrics["macro_precision"])
            history["val_macro_recall"].append(val_metrics["macro_recall"])
            history["val_weighted_f1"].append(val_metrics["weighted_f1"])
            history["lr"].append(warmup_lr)

            is_best = (val_metrics["macro_f1"] > best_macro_f1) or (
                np.isclose(val_metrics["macro_f1"], best_macro_f1)
                and val_metrics["accuracy"] > best_acc
            )
            if is_best:
                best_macro_f1 = val_metrics["macro_f1"]
                best_acc = val_metrics["accuracy"]
                best_epoch = epoch
                best_val_metrics = copy.deepcopy(val_metrics)
                best_state_dict = copy.deepcopy(model.state_dict())

            status_str = f"★ BEST (F1: {best_macro_f1:.4f})" if is_best else ""
            log(f"{epoch:<8}{'Warmup':<12}{train_loss:<13.4f}{train_acc:<12.2f}%"
                f"{val_metrics['loss']:<12.4f}{val_metrics['accuracy']:<11.2f}%"
                f"{val_metrics['macro_f1']:<14.4f}{status_str}")

    # Stage 2: Backbone fine-tuning (all layers unfrozen)
    if finetune_epochs > 0:
        unfreeze_backbone(model, mt)
        param_groups = get_parameter_groups(
            model, mt, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay
        )
        optimizer_ft = AdamW(param_groups)
        scheduler = CosineAnnealingLR(optimizer_ft, T_max=finetune_epochs, eta_min=1e-6)

        for stage2_epoch in range(1, finetune_epochs + 1):
            epoch = warmup_epochs + stage2_epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer_ft, device
            )
            val_metrics, _, _, _ = evaluate_split(
                model, val_loader, criterion, device, active_class_names
            )

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

            is_best = (val_metrics["macro_f1"] > best_macro_f1) or (
                np.isclose(val_metrics["macro_f1"], best_macro_f1)
                and val_metrics["accuracy"] > best_acc
            )
            if is_best:
                best_macro_f1 = val_metrics["macro_f1"]
                best_acc = val_metrics["accuracy"]
                best_epoch = epoch
                best_val_metrics = copy.deepcopy(val_metrics)
                best_state_dict = copy.deepcopy(model.state_dict())

            status_str = f"★ BEST (F1: {best_macro_f1:.4f})" if is_best else ""
            log(f"{epoch:<8}{'FineTune':<12}{train_loss:<13.4f}{train_acc:<12.2f}%"
                f"{val_metrics['loss']:<12.4f}{val_metrics['accuracy']:<11.2f}%"
                f"{val_metrics['macro_f1']:<14.4f}{status_str}")

    total_time = time.time() - start_total_time
    log("-" * 86)
    log(f" TRAINING COMPLETE in {total_time:.1f}s ({total_time / 60.0:.2f} min).")
    log(f" Best Checkpoint: Epoch {best_epoch} | "
        f"Validation Macro-F1 = {best_macro_f1:.4f} | "
        f"Validation Accuracy = {best_acc:.2f}%")
    log("=" * 86)

    # Save Best and Last Checkpoints
    # NOTE: These files are intentionally excluded from the git repository.
    # The checkpoint directory is local only.
    best_ckpt_path = checkpoint_dir / f"best_{mt}.pth"
    last_ckpt_path = checkpoint_dir / f"last_{mt}.pth"

    torch.save({
        "epoch": best_epoch,
        "state_dict": best_state_dict,
        "model_type": mt,
        "val_macro_f1": best_macro_f1,
        "val_accuracy": best_acc,
        "val_metrics": best_val_metrics,
        "history": history,
    }, best_ckpt_path)
    log(f" Saved best checkpoint to: {best_ckpt_path}")

    torch.save({
        "epoch": total_epochs,
        "state_dict": model.state_dict(),
        "model_type": mt,
        "history": history,
    }, last_ckpt_path)
    log(f" Saved last checkpoint to: {last_ckpt_path}")

    # Restore best weights into model before returning
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
        "confusion_matrix": best_val_metrics["confusion_matrix"],
    }

    return model, history, best_val_summary


# ──────────────────────────────────────────────────────────────────────────────
# Unified deep-learning evaluation pipeline (checkpoint-optional)
# ──────────────────────────────────────────────────────────────────────────────

class DeepLearningPipeline:
    """
    Unified execution pipeline for ResNet-50, MRSCAtt, ViT-B/16, and EfficientNet-B3.

    Supports two evaluation modes:
    1. recompute_from_predictions()  — Fast (< 1 s): loads stored prediction arrays
       and recomputes canonical metrics. No checkpoint or GPU required.
    2. evaluate_model()              — Live: loads checkpoint, runs forward pass on
       the dataset. Requires checkpoint file and GPU.

    NOTE: The repository intentionally excludes trained model checkpoint files
    (.pth) because of their large size. Use recompute_from_predictions() for
    result verification from stored artifacts.
    """

    BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
        "resnet50": {
            "display_name": "ResNet-50 (Deep Learning Baseline)",
            "checkpoint": config.MODELS_RESNET50_DIR / "best_resnet50.pth",
            "resolution": (224, 224),
            "benchmark_acc": 77.78,
            "benchmark_macro_f1": 0.6631,
            "predictions_dir": config.RESULTS_RESNET50_DIR / "predictions",
        },
        "mrscatt": {
            "display_name": "MRSCAtt (Spatial & Channel Attention)",
            "checkpoint": config.MODELS_MRSCATT_DIR / "best_mrscatt.pth",
            "resolution": (224, 224),
            "benchmark_acc": 64.29,
            "benchmark_macro_f1": 0.5851,
            "predictions_dir": config.RESULTS_MRSCATT_DIR / "predictions",
        },
        "vit": {
            "display_name": "ViT-B/16 (Vision Transformer)",
            "checkpoint": config.MODELS_VIT_DIR / "best_vit.pth",
            "resolution": (224, 224),
            "benchmark_acc": 73.26,
            "benchmark_macro_f1": 0.6536,
            "predictions_dir": config.RESULTS_VIT_DIR / "predictions",
        },
        "efficientnet_b3": {
            "display_name": "EfficientNet-B3 (Champion Model)",
            "checkpoint": config.MODELS_EFFICIENTNET_B3_DIR / "best_efficientnet_b3.pth",
            "resolution": (300, 300),
            "benchmark_acc": 80.61,
            "benchmark_macro_f1": 0.7083,
            "predictions_dir": config.RESULTS_EFFICIENTNET_B3_DIR / "predictions",
        },
    }

    def __init__(self):
        from deep_learning.datasets import (
            load_split_records,
            load_synset_mapping,
            build_active_class_mappings,
        )
        self.synset_mapping = load_synset_mapping()
        self._splits = {
            "train": load_split_records(config.TRAIN_LABELS_PATH),
            "val": load_split_records(config.VAL_LABELS_PATH),
            "test": load_split_records(config.TEST_LABELS_PATH),
        }
        (
            self.orig_to_active,
            self.active_to_orig,
            self.active_class_names,
            self.active_classes,
        ) = build_active_class_mappings(self._splits["train"], self.synset_mapping)

    def _clean_key(self, model_key: str) -> str:
        k = model_key.lower().replace("-", "_").replace(" ", "_")
        if k == "efficientnet":
            k = "efficientnet_b3"
        if k not in self.BENCHMARK_CONFIGS:
            raise ValueError(
                f"Unknown model: '{model_key}'. "
                f"Choose from {list(self.BENCHMARK_CONFIGS.keys())}"
            )
        return k

    def build_model(self, model_key: str, pretrained: bool = False) -> nn.Module:
        """Construct architecture without loading saved weights."""
        mt = self._clean_key(model_key)
        if mt == "resnet50":
            from deep_learning.models.resnet50 import build_resnet50_baseline
            model, _ = build_resnet50_baseline(num_classes=24, pretrained=pretrained)
        elif mt == "mrscatt":
            from deep_learning.models.mrscatt import build_mrscatt_model
            model, _ = build_mrscatt_model(num_classes=24, pretrained=pretrained)
        elif mt == "vit":
            from deep_learning.models.vit import build_vit_baseline
            model, _ = build_vit_baseline(num_classes=24, pretrained=pretrained)
        elif mt == "efficientnet_b3":
            from deep_learning.models.efficientnet_b3 import build_efficientnet_b3_baseline
            model, _ = build_efficientnet_b3_baseline(num_classes=24, pretrained=pretrained)
        else:
            raise ValueError(f"Unreachable: {mt}")
        return model

    def load_checkpoint(
        self,
        model_key: str,
        checkpoint_path: Optional[Path] = None,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Instantiate architecture and load saved checkpoint."""
        mt = self._clean_key(model_key)
        ckpt_path = checkpoint_path or self.BENCHMARK_CONFIGS[mt]["checkpoint"]
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "NOTE: Checkpoint files are intentionally excluded from the repository. "
                "Use recompute_from_predictions() for checkpoint-free verification."
            )
        model = self.build_model(mt, pretrained=False)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        sd = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(sd)
        return model, checkpoint

    def get_dataloader(
        self,
        model_key: str,
        split: str = "test",
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> DataLoader:
        """Construct DataLoader for specified model and split."""
        mt = self._clean_key(model_key)
        cfg = self.BENCHMARK_CONFIGS[mt]
        resolution = cfg["resolution"]

        from deep_learning.datasets import MarsRoverDataset
        from deep_learning.transforms import get_eval_transforms

        dataset = MarsRoverDataset(
            records=self._splits[split],
            calibrated_dir=config.CALIBRATED_IMG_DIR,
            original_to_active=self.orig_to_active,
            transform=get_eval_transforms(resolution),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def recompute_from_predictions(self, model_key: str) -> Dict[str, Any]:
        """
        Checkpoint-free verification: recompute canonical test metrics directly
        from stored prediction arrays (< 1 s, no GPU or checkpoint required).

        This is the recommended mode when checkpoint files are not available.
        """
        mt = self._clean_key(model_key)
        cfg = self.BENCHMARK_CONFIGS[mt]
        pred_dir = cfg["predictions_dir"]

        preds_path = pred_dir / "test_predictions.npy"
        labels_path = pred_dir / "test_labels.npy"

        if not preds_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Stored predictions not found in {pred_dir}.\n"
                f"Expected: test_predictions.npy and test_labels.npy"
            )

        print(f"  Loading saved test predictions for {cfg['display_name']}...")
        y_true = np.load(labels_path)
        y_pred = np.load(preds_path)

        from deep_learning.evaluation import recompute_metrics_from_predictions
        m = recompute_metrics_from_predictions(
            y_true, y_pred, active_class_names=self.active_class_names
        )
        m["model_key"] = mt
        m["display_name"] = cfg["display_name"]
        m["benchmark_acc"] = cfg["benchmark_acc"]
        m["benchmark_macro_f1"] = cfg["benchmark_macro_f1"]
        m["evaluation_split"] = "test"
        return m

    def evaluate_model(
        self,
        model_key: str,
        split: str = "test",
        device: Optional[torch.device] = None,
        generate_artifacts: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate saved model checkpoint on live dataset split (requires .pth checkpoint).
        Optionally regenerate all result artifacts.
        """
        mt = self._clean_key(model_key)
        from deep_learning.utils import (
            get_device,
            plot_confusion_matrix,
            plot_normalized_confusion_matrix,
            plot_per_class_f1_barchart,
            save_classification_report,
        )
        dev = device or get_device()

        print(f"  Running model inference for {self.BENCHMARK_CONFIGS[mt]['display_name']} on {split} split...")
        model, _ = self.load_checkpoint(mt)
        model.to(dev)
        dataloader = self.get_dataloader(mt, split=split)

        metrics, all_preds, all_targets, all_probs = evaluate_split(
            model=model,
            dataloader=dataloader,
            criterion=None,
            device=dev,
            active_class_names=self.active_class_names,
        )
        cfg = self.BENCHMARK_CONFIGS[mt]
        metrics["model_key"] = mt
        metrics["display_name"] = cfg["display_name"]
        metrics["benchmark_acc"] = cfg["benchmark_acc"]
        metrics["benchmark_macro_f1"] = cfg["benchmark_macro_f1"]
        metrics["evaluation_split"] = split

        if generate_artifacts:
            import json
            dir_map = {
                "resnet50": config.RESULTS_RESNET50_DIR,
                "mrscatt": config.RESULTS_MRSCATT_DIR,
                "vit": config.RESULTS_VIT_DIR,
                "efficientnet_b3": config.RESULTS_EFFICIENTNET_B3_DIR,
            }
            out_dir = dir_map[mt]
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
                save_path=cm_dir / f"{split}_confusion_matrix.png",
            )
            plot_normalized_confusion_matrix(
                cm=cm,
                class_labels=class_labels,
                title=f"{cfg['display_name']} — Normalized Confusion Matrix ({split.capitalize()} Split)",
                save_path=cm_dir / f"{split}_confusion_matrix_normalized.png",
            )

            # Per-class F1 bar chart
            plot_per_class_f1_barchart(
                per_class_dict=metrics["per_class"],
                save_path=plots_dir / f"per_class_f1_{split}_barchart.png",
                title=f"{cfg['display_name']} — Per-Class F1 Scores ({split.capitalize()})",
            )

            # Classification reports
            save_classification_report(
                per_class_dict=metrics["per_class"],
                overall_metrics=metrics,
                csv_path=reports_dir / f"{split}_classification_report.csv",
                txt_path=reports_dir / f"{split}_classification_report.txt",
            )

            # Final metrics JSON
            with open(out_dir / "final_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        return metrics

    def evaluate_all(
        self,
        fast: bool = False,
        generate_artifacts: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all 4 deep learning models.

        fast=True  (default for --verify-results): recomputes from stored prediction arrays.
        fast=False: loads checkpoints and runs live inference.
        """
        results = []
        for key in ["resnet50", "mrscatt", "vit", "efficientnet_b3"]:
            if fast or not generate_artifacts:
                res = self.recompute_from_predictions(key)
            else:
                res = self.evaluate_model(key, split="test", generate_artifacts=generate_artifacts)
            results.append(res)
        return results


# Global singleton instance
dl_pipeline = DeepLearningPipeline()
