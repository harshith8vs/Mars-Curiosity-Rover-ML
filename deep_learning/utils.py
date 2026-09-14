"""Deep learning utilities for seeds, devices, directory initialization, and visualization."""

import json
import os
from pathlib import Path
import random
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import config


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def get_device() -> torch.device:
    """Detect and return the best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def ensure_phase5a_dirs() -> None:
    """Create all required Phase 5A output directories if they do not exist."""
    dirs = [
        config.PHASE5A_RESULTS_DIR,
        config.PHASE5A_REPORTS_DIR,
        config.PHASE5A_CONFUSION_DIR,
        config.PHASE5A_PLOTS_DIR,
        config.PHASE5A_HISTORIES_DIR,
        config.PHASE5A_METRICS_DIR,
        config.PHASE5A_MODELS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def ensure_phase5b_dirs() -> None:
    """Create all required Phase 5B output directories if they do not exist."""
    dirs = [
        config.PHASE5B_RESULTS_DIR,
        config.PHASE5B_REPORTS_DIR,
        config.PHASE5B_CONFUSION_DIR,
        config.PHASE5B_PREDICTIONS_DIR,
        config.PHASE5B_PLOTS_DIR,
        config.PHASE5B_LOGS_DIR,
        config.PHASE5B_METRICS_DIR,
        config.PHASE5B_MODELS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def ensure_phase5c_dirs() -> None:
    """Create all required Phase 5C output directories if they do not exist."""
    dirs = [
        config.PHASE5C_RESULTS_DIR,
        config.PHASE5C_REPORTS_DIR,
        config.PHASE5C_CONFUSION_DIR,
        config.PHASE5C_PREDICTIONS_DIR,
        config.PHASE5C_PLOTS_DIR,
        config.PHASE5C_LOGS_DIR,
        config.PHASE5C_METRICS_DIR,
        config.PHASE5C_MODELS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def ensure_phase5d_dirs() -> None:
    """Create all required Phase 5D output directories if they do not exist."""
    dirs = [
        config.PHASE5D_RESULTS_DIR,
        config.PHASE5D_REPORTS_DIR,
        config.PHASE5D_CONFUSION_DIR,
        config.PHASE5D_PREDICTIONS_DIR,
        config.PHASE5D_PLOTS_DIR,
        config.PHASE5D_LOGS_DIR,
        config.PHASE5D_METRICS_DIR,
        config.PHASE5D_MODELS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def plot_learning_curves(history: Dict[str, List[float]], save_dir: Path) -> None:
    """Generate and save separate learning curves for loss, accuracy, and validation Macro-F1."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss curve
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"], "s-", label="Validation Loss", color="#d62728", linewidth=2)
    plt.title("ResNet-50: Training & Validation Loss vs Epoch", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    plt.savefig(save_dir / "training_loss_curve.png", bbox_inches="tight")
    plt.close()
    
    # Accuracy curve
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(epochs, history["val_acc"], "s-", label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    plt.title("ResNet-50: Training & Validation Accuracy vs Epoch", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "training_accuracy_curve.png", bbox_inches="tight")
    plt.close()
    
    # Validation Macro-F1 curve
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(epochs, history["val_macro_f1"], "d-", label="Validation Macro-F1", color="#9467bd", linewidth=2.2)
    best_idx = int(np.argmax(history["val_macro_f1"]))
    best_epoch = epochs[best_idx]
    best_f1 = history["val_macro_f1"][best_idx]
    plt.scatter([best_epoch], [best_f1], color="#d62728", s=100, zorder=5, label=f"Best (Epoch {best_epoch}: {best_f1:.4f})")
    plt.title("ResNet-50: Validation Macro-F1 vs Epoch (Primary Selection Metric)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Macro-F1 Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "validation_macro_f1_curve.png", bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: List[str],
    title: str,
    save_path: Path
) -> None:
    """Plot and save a high-resolution confusion matrix heatmap."""
    plt.figure(figsize=(14, 12), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Sample Count"},
        linewidths=0.5,
        linecolor="#e0e0e0"
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted NASA Class", fontsize=12, labelpad=10)
    plt.ylabel("True NASA Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_normalized_confusion_matrix(
    cm: np.ndarray,
    class_labels: List[str],
    title: str,
    save_path: Path
) -> None:
    """Plot and save a normalized confusion matrix heatmap."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(14, 12), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Normalized Recall Rate"},
        linewidths=0.5,
        linecolor="#e0e0e0"
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted NASA Class", fontsize=12, labelpad=10)
    plt.ylabel("True NASA Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_per_class_f1_barchart(
    per_class_dict: Dict[int, Dict[str, object]],
    save_path: Path,
    title: str = "ResNet-50: Final Test Per-Class F1 Scores"
) -> None:
    """Plot bar chart of per-class F1 scores."""
    classes = []
    f1_scores = []
    supports = []
    
    for c_idx in sorted(per_class_dict.keys()):
        item = per_class_dict[c_idx]
        classes.append(f"{item['class_name']} (N={item['support']})")
        f1_scores.append(item["f1_score"])
        supports.append(item["support"])
        
    y_pos = np.arange(len(classes))
    
    plt.figure(figsize=(12, 10), dpi=300)
    bars = plt.barh(y_pos, f1_scores, color="#1f77b4", edgecolor="#0e446c", alpha=0.85)
    
    # Highlight zero-support or zero-F1 classes
    for i, bar in enumerate(bars):
        if supports[i] == 0:
            bar.set_color("#d3d3d3")
        elif f1_scores[i] == 0.0:
            bar.set_color("#ff9896")
            
    plt.yticks(y_pos, classes, fontsize=9)
    plt.xlabel("F1 Score", fontsize=11)
    plt.xlim(0.0, 1.05)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    
    # Add text labels on bars
    for i, v in enumerate(f1_scores):
        if supports[i] > 0:
            plt.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color="#333333")
        else:
            plt.text(0.02, i, "N=0 in split", va="center", fontsize=8, color="#777777", fontstyle="italic")
            
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_classification_report(
    per_class_dict: Dict[int, Dict[str, object]],
    overall_metrics: Dict[str, object],
    csv_path: Path,
    txt_path: Path
) -> None:
    """Export classification report as structured CSV and clean monospace TXT."""
    rows = []
    for c_idx in sorted(per_class_dict.keys()):
        item = per_class_dict[c_idx]
        rows.append({
            "Active Index": item["active_index"],
            "Class Name": item["class_name"],
            "Precision": f"{item['precision']:.4f}",
            "Recall": f"{item['recall']:.4f}",
            "F1-Score": f"{item['f1_score']:.4f}",
            "Support": item["support"]
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    acc = overall_metrics.get("accuracy", overall_metrics.get("validation_accuracy", 0.0))
    m_prec = overall_metrics.get("macro_precision", overall_metrics.get("validation_macro_precision", 0.0))
    m_rec = overall_metrics.get("macro_recall", overall_metrics.get("validation_macro_recall", 0.0))
    m_f1 = overall_metrics.get("macro_f1", overall_metrics.get("validation_macro_f1", 0.0))
    w_prec = overall_metrics.get("weighted_precision", overall_metrics.get("validation_weighted_precision", 0.0))
    w_rec = overall_metrics.get("weighted_recall", overall_metrics.get("validation_weighted_recall", 0.0))
    w_f1 = overall_metrics.get("weighted_f1", overall_metrics.get("validation_weighted_f1", 0.0))
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'Active Idx':<12}{'Class Name':<28}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<8}\n")
        f.write("-" * 84 + "\n")
        for r in rows:
            f.write(f"{r['Active Index']:<12}{r['Class Name']:<28}{r['Precision']:<12}{r['Recall']:<12}{r['F1-Score']:<12}{r['Support']:<8}\n")
        f.write("-" * 84 + "\n")
        f.write(f"{'Overall Accuracy':<40}{acc:.2f}%\n")
        f.write(f"{'Macro Average':<40}P={m_prec:.4f}, R={m_rec:.4f}, F1={m_f1:.4f}\n")
        f.write(f"{'Weighted Average':<40}P={w_prec:.4f}, R={w_rec:.4f}, F1={w_f1:.4f}\n")
