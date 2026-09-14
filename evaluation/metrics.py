"""Evaluation metrics calculation and confusion matrix visualization."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate accuracy, macro precision/recall/F1, and weighted F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def generate_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str],
    total_classes: int = 25
) -> pd.DataFrame:
    """Generate pandas DataFrame classification report preserving all defined class IDs."""
    labels = list(range(total_classes))
    target_names = [f"{c:02d}_{class_names.get(c, 'unknown').replace(' ', '_')}" for c in labels]
    
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    return pd.DataFrame(report_dict).transpose()


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str],
    title: str,
    save_path: Path,
    cmap: str = "Blues",
    total_classes: int = 25
) -> np.ndarray:
    """Generate and save confusion matrix heatmap."""
    labels = list(range(total_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    tick_labels = [f"{c:02d} {class_names.get(c, 'unknown')}" for c in labels]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=tick_labels,
        yticklabels=tick_labels
    )
    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return cm


def plot_and_save_normalized_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str],
    title: str,
    save_path: Path,
    total_classes: int = 25
) -> np.ndarray:
    """Generates and saves a normalized confusion matrix heatmap."""
    labels = list(range(total_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    tick_labels = [f"{c:02d} {class_names.get(c, 'unknown')}" for c in labels]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"label": "Normalized Recall Rate"}
    )
    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return cm_norm
