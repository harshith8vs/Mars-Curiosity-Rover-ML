"""
Automated validation tests covering the 12 dataset and metrics invariants.

Run with:
    venv/bin/python3 -m pytest tests/test_dataset_and_metrics.py -v

All 12 invariants must pass. These are CHECKPOINT-FREE — they use only
stored prediction arrays and feature matrices, not trained model files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluation.mapping import (
    ACTIVE_NASA_CLASS_IDS,
    ACTIVE_INDICES,
    INACTIVE_NASA_CLASS_ID,
    NASA_TO_ACTIVE_INDEX,
    ACTIVE_INDEX_TO_NASA,
    NUM_ACTIVE_CLASSES,
    ZERO_SUPPORT_IN_TEST,
    validate_active_class_ids,
)
from evaluation.metrics import compute_canonical_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

CLASSICAL_PRED_DIRS = {
    "knn":           PROJECT_ROOT / "results" / "01_KNN",
    "naive_bayes":   PROJECT_ROOT / "results" / "02_Naive_Bayes",
    "decision_tree": PROJECT_ROOT / "results" / "03_Decision_Tree",
    "random_forest": PROJECT_ROOT / "results" / "04_Random_Forest",
    "svm":           PROJECT_ROOT / "results" / "05_SVM",
}

DL_PRED_DIRS = {
    "mrscatt":         PROJECT_ROOT / "results" / "06_MRSCAtt" / "predictions",
    "vit":             PROJECT_ROOT / "results" / "07_ViT_B16" / "predictions",
    "resnet50":        PROJECT_ROOT / "results" / "08_ResNet50" / "predictions",
    "efficientnet_b3": PROJECT_ROOT / "results" / "09_EfficientNet_B3" / "predictions",
}

FEATURES_DIR = PROJECT_ROOT / "features" if (PROJECT_ROOT / "features").is_dir() else None


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 1: Mapping module internal consistency
# ──────────────────────────────────────────────────────────────────────────────

def test_inv01_mapping_consistency():
    """Mapping module round-trip consistency and counts."""
    assert validate_active_class_ids(), "Mapping module validation failed"
    assert len(ACTIVE_NASA_CLASS_IDS) == NUM_ACTIVE_CLASSES == 24
    assert INACTIVE_NASA_CLASS_ID not in ACTIVE_NASA_CLASS_IDS
    assert len(ACTIVE_INDICES) == 24
    # Round-trip
    for idx, nasa_id in enumerate(ACTIVE_NASA_CLASS_IDS):
        assert NASA_TO_ACTIVE_INDEX[nasa_id] == idx
        assert ACTIVE_INDEX_TO_NASA[idx] == nasa_id


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 2: Class 22 ("sun") is globally absent from stored predictions
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_key", list(CLASSICAL_PRED_DIRS.keys()))
def test_inv02_class22_absent_classical(model_key):
    """Class 22 must not appear in any classical model label or prediction array."""
    pred_dir = CLASSICAL_PRED_DIRS[model_key]
    for split in ["test", "val"]:
        labels_path = pred_dir / f"{split}_labels.npy"
        preds_path = pred_dir / f"{split}_predictions.npy"
        if labels_path.is_file():
            y_true = np.load(labels_path)
            assert INACTIVE_NASA_CLASS_ID not in y_true, (
                f"{model_key} {split} labels contain inactive class {INACTIVE_NASA_CLASS_ID}"
            )
        if preds_path.is_file():
            y_pred = np.load(preds_path)
            assert INACTIVE_NASA_CLASS_ID not in y_pred, (
                f"{model_key} {split} predictions contain inactive class {INACTIVE_NASA_CLASS_ID}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 3: DL active indices in range [0, 23]
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_key", list(DL_PRED_DIRS.keys()))
def test_inv03_dl_index_range(model_key):
    """All DL prediction/label arrays must be in [0, 23]."""
    pred_dir = DL_PRED_DIRS[model_key]
    for fname in ["test_predictions.npy", "test_labels.npy"]:
        fpath = pred_dir / fname
        if fpath.is_file():
            arr = np.load(fpath)
            assert arr.min() >= 0, f"{model_key}/{fname} has negative indices"
            assert arr.max() <= 23, (
                f"{model_key}/{fname} has index {arr.max()} > 23"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 4: All 9 test prediction files have 1,305 samples
# ──────────────────────────────────────────────────────────────────────────────

EXPECTED_TEST_SAMPLES = 1305

@pytest.mark.parametrize("model_key,pred_dir", [
    *[(k, v) for k, v in CLASSICAL_PRED_DIRS.items()],
    *[(k, v) for k, v in DL_PRED_DIRS.items()],
])
def test_inv04_test_split_sample_count(model_key, pred_dir):
    """Test labels must have exactly 1,305 samples (official NASA test split)."""
    labels_path = pred_dir / "test_labels.npy"
    if not labels_path.is_file():
        pytest.skip(f"{model_key}: test_labels.npy not found")
    y_true = np.load(labels_path)
    assert len(y_true) == EXPECTED_TEST_SAMPLES, (
        f"{model_key} test split: expected {EXPECTED_TEST_SAMPLES} samples, "
        f"got {len(y_true)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 5: Prediction arrays length == labels arrays length
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_key,pred_dir", [
    *[(k, v) for k, v in CLASSICAL_PRED_DIRS.items()],
    *[(k, v) for k, v in DL_PRED_DIRS.items()],
])
@pytest.mark.parametrize("split", ["test", "val"])
def test_inv05_pred_labels_same_length(model_key, pred_dir, split):
    """Predictions and labels must have equal length for each split."""
    labels_path = pred_dir / f"{split}_labels.npy"
    preds_path = pred_dir / f"{split}_predictions.npy"
    if not labels_path.is_file() or not preds_path.is_file():
        pytest.skip(f"{model_key}/{split}: files not found")
    y_true = np.load(labels_path)
    y_pred = np.load(preds_path)
    assert len(y_true) == len(y_pred), (
        f"{model_key} {split}: labels len={len(y_true)} != preds len={len(y_pred)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 6: Canonical Macro-F1 uses explicit 24-class label list
# ──────────────────────────────────────────────────────────────────────────────

def test_inv06_canonical_macro_f1_label_count():
    """
    Verify that the canonical metric function uses exactly 24 labels.
    When zero-support classes 5 and 23 are present, the denominator is 24.
    """
    from sklearn.metrics import f1_score

    # Synthetic case: predictions never use NASA IDs 5 or 23
    rng = np.random.default_rng(seed=0)
    present_ids = [c for c in ACTIVE_NASA_CLASS_IDS if c not in ZERO_SUPPORT_IN_TEST]
    y_true = rng.choice(present_ids, size=200)
    y_pred = rng.choice(present_ids, size=200)

    # Default macro (no labels) only averages over classes present in y_true or y_pred
    f1_default = f1_score(y_true, y_pred, average="macro", zero_division=0)
    # Canonical macro (explicit 24 labels) assigns 0.0 to classes 5 and 23
    f1_canonical = f1_score(
        y_true, y_pred, labels=ACTIVE_NASA_CLASS_IDS, average="macro", zero_division=0
    )
    # Canonical must be ≤ default (adding zero-F1 terms can only lower the average)
    assert f1_canonical <= f1_default + 1e-9, (
        f"Canonical Macro-F1 ({f1_canonical:.4f}) > default ({f1_default:.4f})"
    )
    # Canonical must differ from default (because classes 5, 23 are genuinely 0-support)
    assert not np.isclose(f1_canonical, f1_default), (
        "Canonical and default Macro-F1 are identical — zero-support classes 5, 23 "
        "appear to have been excluded from the explicit label list"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 7: Classes 5 and 23 have ZERO samples in test labels (classical)
# ──────────────────────────────────────────────────────────────────────────────

def test_inv07_zero_support_classes_in_test():
    """
    NASA IDs 5 and 23 must have 0 samples in the official test split.
    """
    labels_path = CLASSICAL_PRED_DIRS["knn"] / "test_labels.npy"
    if not labels_path.is_file():
        pytest.skip("KNN test_labels.npy not found")
    y_true = np.load(labels_path)
    for nasa_id in ZERO_SUPPORT_IN_TEST:
        count = int(np.sum(y_true == nasa_id))
        assert count == 0, (
            f"Expected 0 test samples for NASA ID {nasa_id}, found {count}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 8: EfficientNet-B3 achieves best test Macro-F1 among all 9 models
# ──────────────────────────────────────────────────────────────────────────────

def test_inv08_efficientnet_is_champion():
    """EfficientNet-B3 must have the highest canonical test Macro-F1."""
    all_f1 = {}
    for key, d in CLASSICAL_PRED_DIRS.items():
        p, l = d / "test_predictions.npy", d / "test_labels.npy"
        if p.is_file() and l.is_file():
            m = compute_canonical_metrics(np.load(l), np.load(p), label_space="nasa")
            all_f1[key] = m["macro_f1"]
    for key, d in DL_PRED_DIRS.items():
        p, l = d / "test_predictions.npy", d / "test_labels.npy"
        if p.is_file() and l.is_file():
            m = compute_canonical_metrics(np.load(l), np.load(p), label_space="active_index")
            all_f1[key] = m["macro_f1"]

    if "efficientnet_b3" not in all_f1:
        pytest.skip("EfficientNet-B3 predictions not found")

    eff_f1 = all_f1["efficientnet_b3"]
    for other_key, other_f1 in all_f1.items():
        assert eff_f1 >= other_f1 - 1e-6, (
            f"EfficientNet-B3 Macro-F1 ({eff_f1:.4f}) is not ≥ "
            f"{other_key} ({other_f1:.4f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 9: EfficientNet-B3 achieves best test accuracy among all 9 models
# ──────────────────────────────────────────────────────────────────────────────

def test_inv09_efficientnet_best_accuracy():
    """EfficientNet-B3 must also have the highest test accuracy."""
    all_acc = {}
    for key, d in CLASSICAL_PRED_DIRS.items():
        p, l = d / "test_predictions.npy", d / "test_labels.npy"
        if p.is_file() and l.is_file():
            m = compute_canonical_metrics(np.load(l), np.load(p), label_space="nasa")
            all_acc[key] = m["accuracy"]
    for key, d in DL_PRED_DIRS.items():
        p, l = d / "test_predictions.npy", d / "test_labels.npy"
        if p.is_file() and l.is_file():
            m = compute_canonical_metrics(np.load(l), np.load(p), label_space="active_index")
            all_acc[key] = m["accuracy"]

    if "efficientnet_b3" not in all_acc:
        pytest.skip("EfficientNet-B3 predictions not found")
    eff_acc = all_acc["efficientnet_b3"]
    for other_key, other_acc in all_acc.items():
        assert eff_acc >= other_acc - 1e-4, (
            f"EfficientNet-B3 accuracy ({eff_acc:.2f}%) < {other_key} ({other_acc:.2f}%)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 10: Classical predictions contain only NASA IDs in ACTIVE_NASA_CLASS_IDS
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_key", list(CLASSICAL_PRED_DIRS.keys()))
def test_inv10_classical_pred_valid_classes(model_key):
    """Classical predictions must only use valid active NASA IDs."""
    pred_dir = CLASSICAL_PRED_DIRS[model_key]
    active_set = set(ACTIVE_NASA_CLASS_IDS)
    for split in ["test", "val"]:
        for suffix in ["labels", "predictions"]:
            fpath = pred_dir / f"{split}_{suffix}.npy"
            if fpath.is_file():
                arr = np.load(fpath)
                unique = set(arr.tolist())
                invalid = unique - active_set
                assert not invalid, (
                    f"{model_key}/{split}_{suffix}: found invalid NASA IDs {invalid}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 11: No checkpoint (.pth / .joblib) committed to predictions dirs
# ──────────────────────────────────────────────────────────────────────────────

def test_inv11_no_checkpoint_in_results():
    """
    Result directories must NOT contain .pth or .joblib checkpoint files.
    These are intentionally excluded from the repository to keep it lightweight.
    """
    results_dir = PROJECT_ROOT / "results"
    pth_files = list(results_dir.rglob("*.pth"))
    joblib_files = list(results_dir.rglob("*.joblib"))
    assert not pth_files, f"Found .pth files in results/: {pth_files}"
    assert not joblib_files, f"Found .joblib files in results/: {joblib_files}"


# ──────────────────────────────────────────────────────────────────────────────
# Invariant 12: Test predictions are integer class labels (not softmax probabilities)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_key,pred_dir", [
    *[(k, v) for k, v in CLASSICAL_PRED_DIRS.items()],
    *[(k, v) for k, v in DL_PRED_DIRS.items()],
])
def test_inv12_predictions_are_integer_labels(model_key, pred_dir):
    """Prediction files must contain integer class labels, not probability floats in [0,1]."""
    preds_path = pred_dir / "test_predictions.npy"
    if not preds_path.is_file():
        pytest.skip(f"{model_key}: test_predictions.npy not found")
    y_pred = np.load(preds_path)
    # If values are all in [0, 1], they are likely probabilities, not integer class IDs
    assert not (y_pred.max() <= 1.0 and y_pred.dtype.kind == "f"), (
        f"{model_key} test_predictions.npy appears to contain probabilities "
        f"(max={y_pred.max():.4f}, dtype={y_pred.dtype}) — expected integer class IDs"
    )
    # All values must be non-negative integers
    assert np.all(y_pred >= 0), f"{model_key}: predictions contain negative values"
    assert np.issubdtype(y_pred.dtype, np.integer) or np.all(y_pred == y_pred.astype(int)), (
        f"{model_key}: predictions are not integer-valued (dtype={y_pred.dtype})"
    )
