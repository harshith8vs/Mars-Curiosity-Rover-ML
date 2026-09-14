"""Class imbalance auditing and balanced class weight computation."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn


def audit_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: Dict[int, str]
) -> pd.DataFrame:
    """Audit class frequencies and proportions across dataset splits."""
    total_images = len(y_train) + len(y_val) + len(y_test)
    records = []

    for cid in range(25):
        cname = class_names.get(cid, f"Class_{cid}")
        tr_cnt = int(np.sum(y_train == cid))
        vl_cnt = int(np.sum(y_val == cid))
        ts_cnt = int(np.sum(y_test == cid))
        tot = tr_cnt + vl_cnt + ts_cnt
        pct = (tot / total_images) * 100.0 if total_images > 0 else 0.0

        records.append({
            "Class ID": cid,
            "Class Name": cname,
            "Train Count": tr_cnt,
            "Validation Count": vl_cnt,
            "Test Count": ts_cnt,
            "Total Count": tot,
            "Percentage (%)": round(pct, 2)
        })

    return pd.DataFrame(records)


def compute_balanced_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights w_c = N / (K * N_c) derived strictly from y_train."""
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)

    N = len(y_train)
    active_classes = np.unique(y_train)
    K = len(active_classes)

    weights: Dict[int, float] = {}
    for c in sorted(active_classes):
        n_c = int(np.sum(y_train == c))
        if n_c > 0:
            weights[int(c)] = float(N / (K * n_c))

    return weights


def get_model_imbalance_support() -> Dict[str, Dict[str, Any]]:
    """
    Inspects actual installed scikit-learn estimator APIs to document supported
    imbalance mechanisms across the 5 planned models.
    
    Returns:
        Dict mapping model name -> capability specifications.
    """
    # Programmatic check of constructor parameters
    knn_params = KNeighborsClassifier().get_params()
    gnb_params = GaussianNB().get_params()
    dt_params = DecisionTreeClassifier().get_params()
    rf_params = RandomForestClassifier().get_params()
    svm_params = SVC().get_params()

    support_matrix = {
        "KNN (KNeighborsClassifier)": {
            "supports_class_weight": "class_weight" in knn_params,
            "supported_mechanism": "Distance weighting (weights='distance')",
            "baseline_config": {"weights": "uniform"},
            "balanced_config": {"weights": "distance"},
            "reason": (
                "class_weight is not supported by this estimator. "
                "KNN predicts based on local neighborhood density rather than a global loss function. "
                "The natural baseline uses uniform neighbor voting (weights='uniform'). "
                "The alternative imbalance-aware mechanism is inverse distance weighting (weights='distance'), "
                "giving closer neighbors greater voting weight without synthetic sample duplication."
            )
        },
        "Gaussian Naive Bayes (GaussianNB)": {
            "supports_class_weight": "class_weight" in gnb_params,
            "supported_mechanism": "Prior probability adjustment (priors parameter)",
            "baseline_config": {"priors": None},
            "balanced_config": {"priors": "uniform_array"},
            "reason": (
                "class_weight is not supported by this estimator. "
                "GaussianNB models class conditional likelihoods and prior probabilities P(y). "
                "The natural baseline uses empirical frequency priors (priors=None). "
                "The balanced alternative specifies equal uniform prior probabilities across active classes "
                "(priors = np.ones(K) / K), preventing majority classes from overpowering bayesian posterior probabilities."
            )
        },
        "Decision Tree (DecisionTreeClassifier)": {
            "supports_class_weight": "class_weight" in dt_params,
            "supported_mechanism": "Loss function sample weighting (class_weight='balanced' or custom dict)",
            "baseline_config": {"class_weight": None},
            "balanced_config": {"class_weight": "balanced"},
            "reason": (
                "class_weight is fully supported by this estimator. "
                "Decision trees weight impurity measures (Gini / Entropy) by class frequency. "
                "The baseline uses natural split frequencies (class_weight=None). "
                "The balanced configuration uses class_weight='balanced' to penalize misclassifications "
                "inversely proportional to class frequencies."
            )
        },
        "Random Forest (RandomForestClassifier)": {
            "supports_class_weight": "class_weight" in rf_params,
            "supported_mechanism": "Ensemble sample weighting (class_weight='balanced' or 'balanced_subsample')",
            "baseline_config": {"class_weight": None},
            "balanced_config": {"class_weight": "balanced"},
            "reason": (
                "class_weight is fully supported by this estimator. "
                "Supports both global balanced weighting (class_weight='balanced') and bootstrap-sample "
                "recomputed weighting (class_weight='balanced_subsample'). Baseline uses class_weight=None."
            )
        },
        "SVM (SVC)": {
            "supports_class_weight": "class_weight" in svm_params,
            "supported_mechanism": "Margin penalty regularization weighting (class_weight='balanced' or custom dict)",
            "baseline_config": {"class_weight": None},
            "balanced_config": {"class_weight": "balanced"},
            "reason": (
                "class_weight is fully supported by this estimator. "
                "SVC scales the soft-margin slack penalty C for each class (C_i = C * weight_i). "
                "The baseline uses uniform penalty C for all classes (class_weight=None). "
                "The balanced configuration applies higher slack penalties to errors on minority classes."
            )
        }
    }

    return support_matrix


def validate_class_weights(weights: Dict[int, float], y_train: np.ndarray) -> bool:
    """Validate that class weights are finite, positive, and match active training classes."""
    active_in_train = set(np.unique(y_train))
    classes_in_weights = set(weights.keys())

    if classes_in_weights != active_in_train:
        raise ValueError(
            f"Class weight key mismatch! Weights keys: {sorted(classes_in_weights)}, "
            f"Active in y_train: {sorted(active_in_train)}"
        )

    for cid, w in weights.items():
        if not np.isfinite(w) or w <= 0.0:
            raise ValueError(f"Invalid weight for class {cid}: {w} (must be finite and > 0)")

    return True
