"""
Canonical evaluation metrics for the NASA Mars Curiosity Rover classification benchmark.

All metric computations use the canonical active-class definitions from evaluation.mapping:
  - Classical ML:    labels=ACTIVE_NASA_CLASS_IDS  (NASA IDs 0-21, 23, 24)
  - Deep Learning:   labels=ACTIVE_INDICES          (contiguous 0-23)

Class 22 ("sun") is GLOBALLY INACTIVE (zero samples in all splits) and is never
included as a label.

NASA IDs 5 ("drill holes") and 23 ("turret") ARE active classes but have
zero ground-truth samples in the official TEST split. They remain in the
Macro-F1 denominator; zero_division=0 assigns F1=0.0 to them, so
averaging is always over the full 24 active classes.
"""

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
    confusion_matrix,
    precision_recall_fscore_support
)

from evaluation.mapping import (
    ACTIVE_NASA_CLASS_IDS,
    ACTIVE_INDICES,
    NUM_ACTIVE_CLASSES,
    NASA_ID_TO_CLASS_NAME,
    ACTIVE_INDEX_TO_CLASS_NAME,
    INACTIVE_NASA_CLASS_ID,
    ZERO_SUPPORT_IN_TEST,
)


# ──────────────────────────────────────────────────────────────────────────────
# Core canonical metric function (single authoritative implementation)
# ──────────────────────────────────────────────────────────────────────────────

def compute_canonical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_space: str = "nasa",
) -> Dict[str, float]:
    """
    Compute accuracy, macro precision/recall/F1, and weighted F1 using explicit
    active-class label lists to ensure consistent 24-class denominator.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground-truth and predicted labels.
    label_space : {"nasa", "active_index"}
        "nasa"         — labels are raw NASA IDs (0-21, 23, 24); used for classical ML.
        "active_index" — labels are remapped DL indices (0-23); used for deep learning.

    Returns
    -------
    dict with keys: accuracy, macro_precision, macro_recall, macro_f1, weighted_f1
    """
    if label_space == "nasa":
        labels = ACTIVE_NASA_CLASS_IDS        # explicit 24-element list of NASA IDs
    elif label_space == "active_index":
        labels = ACTIVE_INDICES               # [0, 1, …, 23]
    else:
        raise ValueError(f"label_space must be 'nasa' or 'active_index', got '{label_space}'")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Legacy wrapper — computes metrics using default sklearn behaviour (no explicit labels).
    Prefer compute_canonical_metrics() for all new code.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_space: str = "nasa",
) -> Dict[int, Dict[str, Union[float, int]]]:
    """
    Compute per-class precision, recall, F1, and support.

    Returns a dict keyed by either NASA ID or DL active index (depending on label_space),
    with values: {precision, recall, f1_score, support, class_name}.
    """
    if label_space == "nasa":
        labels = ACTIVE_NASA_CLASS_IDS
        name_map = NASA_ID_TO_CLASS_NAME
    else:
        labels = ACTIVE_INDICES
        name_map = ACTIVE_INDEX_TO_CLASS_NAME

    per_p, per_r, per_f1, per_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    result = {}
    for i, lbl in enumerate(labels):
        result[lbl] = {
            "class_name": name_map.get(lbl, f"class_{lbl}"),
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1_score": float(per_f1[i]),
            "support": int(per_supp[i]),
        }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Classification report (DataFrame)
# ──────────────────────────────────────────────────────────────────────────────

def generate_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    label_space: str = "nasa",
    total_classes: int = 25,  # kept for legacy call compatibility, ignored
) -> pd.DataFrame:
    """
    Generate a pandas DataFrame classification report over the 24 active classes.

    Class 22 ("sun") is explicitly excluded.  The macro averages in the report
    therefore correspond to the canonical 24-class protocol.

    Parameters
    ----------
    y_true, y_pred : arrays of labels
    class_names : optional dict overriding default names
    label_space : "nasa" or "active_index"
    total_classes : ignored (kept for API backward compatibility)
    """
    if label_space == "nasa":
        labels = ACTIVE_NASA_CLASS_IDS
        default_names = NASA_ID_TO_CLASS_NAME
    else:
        labels = ACTIVE_INDICES
        default_names = ACTIVE_INDEX_TO_CLASS_NAME

    names = class_names if class_names is not None else default_names
    target_names = [
        f"{lbl:02d}_{names.get(lbl, 'unknown').replace(' ', '_')}"
        for lbl in labels
    ]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(report_dict).transpose()
    # Add a note about excluded class
    df.attrs["excluded_class"] = (
        f"Class {INACTIVE_NASA_CLASS_ID} ('{INACTIVE_NASA_CLASS_ID}') excluded — zero samples in all splits."
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrices
# ──────────────────────────────────────────────────────────────────────────────

def _build_cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_space: str,
) -> tuple:
    """Helper returning (cm, labels, tick_labels)."""
    if label_space == "nasa":
        labels = ACTIVE_NASA_CLASS_IDS
        name_map = NASA_ID_TO_CLASS_NAME
    else:
        labels = ACTIVE_INDICES
        name_map = ACTIVE_INDEX_TO_CLASS_NAME

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick_labels = [f"{lbl:02d} {name_map.get(lbl, 'unknown')}" for lbl in labels]
    return cm, labels, tick_labels


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    cmap: str = "Blues",
    label_space: str = "nasa",
    total_classes: int = 25,  # legacy param, ignored
) -> np.ndarray:
    """Generate and save a 24×24 confusion matrix heatmap (raw counts)."""
    cm, labels, tick_labels = _build_cm(y_true, y_pred, label_space)

    if class_names:
        tick_labels = [
            f"{lbl:02d} {class_names.get(lbl, 'unknown')}" for lbl in labels
        ]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=tick_labels, yticklabels=tick_labels
    )
    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
    return cm


def plot_and_save_normalized_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    title: str = "Normalized Confusion Matrix",
    save_path: Optional[Path] = None,
    label_space: str = "nasa",
    total_classes: int = 25,  # legacy param, ignored
) -> np.ndarray:
    """Generate and save a 24×24 normalized confusion matrix heatmap (row-normalized recall)."""
    cm, labels, tick_labels = _build_cm(y_true, y_pred, label_space)

    if class_names:
        tick_labels = [
            f"{lbl:02d} {class_names.get(lbl, 'unknown')}" for lbl in labels
        ]

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=tick_labels, yticklabels=tick_labels,
        cbar_kws={"label": "Normalized Recall Rate"}
    )
    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
    return cm_norm
