"""
Classical machine learning pipeline — training, evaluation, and inference.

Evaluation Protocol
-------------------
Phase A (Model Selection):
  Training split → train models.
  Validation split → select champion configuration per model family.
  Primary metric = Validation Macro-F1.

Phase B (Final Evaluation):
  Test split → evaluate ALL 9 final models once on the same untouched test set.

Metrics use canonical 24-class labels (ACTIVE_NASA_CLASS_IDS) with zero_division=0.
Class 22 ("sun") is excluded globally.
NASA IDs 5 and 23 are active but have zero test-set samples; they remain in the
Macro-F1 denominator (F1=0.0, averaged over all 24 classes).
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

import config
from evaluation.mapping import ACTIVE_NASA_CLASS_IDS, NUM_ACTIVE_CLASSES
from evaluation.metrics import (
    compute_canonical_metrics,
    generate_classification_report_df,
    plot_and_save_confusion_matrix,
    plot_and_save_normalized_confusion_matrix,
)
from models.decision_tree import build_decision_tree_model, train_decision_tree
from models.imbalance import compute_balanced_class_weights
from models.knn import build_knn_model, train_knn
from models.naive_bayes import build_naive_bayes_model, train_naive_bayes
from models.random_forest import build_random_forest_model, train_random_forest
from models.scaling import TrainingScaler
from models.svm import build_svm_model, train_svm


# ──────────────────────────────────────────────────────────────────────────────
# Champion classical configurations
# ──────────────────────────────────────────────────────────────────────────────

CLASSICAL_BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "knn": {
        "config_id": "KNN-4",
        "display_name": "KNN (Scaled+PCA, Distance)",
        "representation": "pca",
        "checkpoint": config.SAVED_MODELS_DIR / "knn" / "KNN-4.joblib",
        # Selection metric: Validation Macro-F1 (explicitly over 24 active classes)
        "val_benchmark_macro_f1": 0.5511,  # canonical 24-class val Macro-F1
        "val_benchmark_acc": 50.85,
        # Legacy benchmark values (used in older comparison tables, default sklearn macro)
        "benchmark_acc": 50.85,
        "benchmark_macro_f1": 0.6613,
    },
    "naive_bayes": {
        "config_id": "NB-2",
        "display_name": "GaussianNB (Scaled, Uniform Priors)",
        "representation": "scaled",
        "checkpoint": config.SAVED_MODELS_DIR / "naive_bayes" / "NB-2.joblib",
        "val_benchmark_macro_f1": 0.1844,
        "val_benchmark_acc": 12.07,
        "benchmark_acc": 12.07,
        "benchmark_macro_f1": 0.2012,
    },
    "decision_tree": {
        "config_id": "DT-2",
        "display_name": "DecisionTree (Unscaled, Balanced)",
        "representation": "raw",
        "checkpoint": config.SAVED_MODELS_DIR / "decision_tree" / "DT-2.joblib",
        "val_benchmark_macro_f1": 0.2690,
        "val_benchmark_acc": 15.30,
        "benchmark_acc": 15.30,
        "benchmark_macro_f1": 0.2690,
    },
    "random_forest": {
        "config_id": "RF-2",
        "display_name": "RandomForest (Unscaled, Balanced)",
        "representation": "raw",
        "checkpoint": config.SAVED_MODELS_DIR / "random_forest" / "RF-2.joblib",
        "val_benchmark_macro_f1": 0.5494,
        "val_benchmark_acc": 66.04,
        "benchmark_acc": 66.04,
        "benchmark_macro_f1": 0.5993,
    },
    "svm": {
        "config_id": "SVM-4",
        "display_name": "SVM (Scaled+PCA, Balanced)",
        "representation": "pca",
        "checkpoint": config.SAVED_MODELS_DIR / "svm" / "SVM-4.joblib",
        "val_benchmark_macro_f1": 0.5594,
        "val_benchmark_acc": 67.01,
        "benchmark_acc": 67.01,
        "benchmark_macro_f1": 0.6393,
    },
}

# Result directories (one per model)
_RESULT_DIRS: Dict[str, Path] = {
    "knn": config.RESULTS_KNN_DIR,
    "naive_bayes": config.RESULTS_NAIVE_BAYES_DIR,
    "decision_tree": config.RESULTS_DECISION_TREE_DIR,
    "random_forest": config.RESULTS_RANDOM_FOREST_DIR,
    "svm": config.RESULTS_SVM_DIR,
}


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ──────────────────────────────────────────────────────────────────────────────

class ClassicalPipeline:
    """
    Manages data loading, preprocessors (TrainingScaler, PCA), evaluation,
    and checkpoint-free verification for classical ML models.

    Scaler and PCA are fitted ONLY on training data (no leakage).
    """

    def __init__(self):
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._scaler: Optional[TrainingScaler] = None
        self._pca: Optional[PCA] = None

    def load_features(self) -> None:
        """Load cached feature matrices and labels from features/ directory."""
        if self._X_train is None:
            print("  Loading feature matrices...")
            self._X_train = np.load(config.FEATURES_DIR / "X_train.npz")["X"]
            self._y_train = np.load(config.FEATURES_DIR / "y_train.npy")
            self._X_val = np.load(config.FEATURES_DIR / "X_val.npz")["X"]
            self._y_val = np.load(config.FEATURES_DIR / "y_val.npy")
            self._X_test = np.load(config.FEATURES_DIR / "X_test.npz")["X"]
            self._y_test = np.load(config.FEATURES_DIR / "y_test.npy")

    def get_preprocessors(self) -> Tuple[TrainingScaler, PCA]:
        """
        Fit TrainingScaler and PCA(n_components=100) EXCLUSIVELY on training data.
        Never fits on validation or test data.
        """
        self.load_features()
        if self._scaler is None:
            self._scaler = TrainingScaler()
            self._scaler.fit(self._X_train)    # fit on train only
        if self._pca is None:
            X_train_scaled = self._scaler.transform(self._X_train)
            self._pca = PCA(n_components=100, random_state=42)
            self._pca.fit(X_train_scaled)       # fit on train only
        return self._scaler, self._pca

    def get_split_data(
        self, split: str = "val", representation: str = "raw"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve features and labels for a split + representation.

        Parameters
        ----------
        split : "train" | "val" | "test"
        representation : "raw" | "scaled" | "pca"
        """
        self.load_features()
        scaler, pca = self.get_preprocessors()

        data_map = {
            "train": (self._X_train, self._y_train),
            "val": (self._X_val, self._y_val),
            "test": (self._X_test, self._y_test),
        }
        if split not in data_map:
            raise ValueError(f"Unknown split: '{split}'. Choose 'train', 'val', or 'test'.")
        X, y = data_map[split]

        if representation == "raw":
            return X, y
        elif representation == "scaled":
            return scaler.transform(X), y
        elif representation == "pca":
            return pca.transform(scaler.transform(X)), y
        else:
            raise ValueError(
                f"Unknown representation: '{representation}'. "
                "Choose 'raw', 'scaled', or 'pca'."
            )

    def recompute_from_predictions(
        self, model_key: str, split: str = "test"
    ) -> Dict[str, Any]:
        """
        Checkpoint-free verification: compute canonical metrics directly from
        stored prediction arrays (no .joblib file required).

        This is the recommended verification path when checkpoint files are absent.
        """
        clean_key = model_key.lower().replace("-", "_").replace(" ", "_")
        if clean_key not in CLASSICAL_BENCHMARK_CONFIGS:
            raise ValueError(
                f"Model '{model_key}' not recognized. "
                f"Available: {list(CLASSICAL_BENCHMARK_CONFIGS.keys())}"
            )
        cfg = CLASSICAL_BENCHMARK_CONFIGS[clean_key]
        out_dir = _RESULT_DIRS[clean_key]

        preds_path = out_dir / f"{split}_predictions.npy"
        labels_path = out_dir / f"{split}_labels.npy"

        if not preds_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Stored {split} predictions not found in {out_dir}.\n"
                f"Expected: {split}_predictions.npy and {split}_labels.npy"
            )

        print(
            f"  Loading saved {split} predictions for "
            f"{cfg['display_name']} ({cfg['config_id']})..."
        )
        y_true = np.load(labels_path)
        y_pred = np.load(preds_path)

        # Use canonical 24-class metrics (NASA IDs for classical models)
        m = compute_canonical_metrics(y_true, y_pred, label_space="nasa")
        return {
            "model_key": clean_key,
            "config_id": cfg["config_id"],
            "display_name": cfg["display_name"],
            "split": split,
            "evaluation_split": split.upper(),
            "num_samples": int(len(y_true)),
            **m,
            "benchmark_acc": cfg["benchmark_acc"],
            "benchmark_macro_f1": cfg["benchmark_macro_f1"],
        }

    def evaluate_model(
        self,
        model_key: str,
        split: str = "test",
        generate_artifacts: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained classical model on a dataset split.
        Loads the .joblib checkpoint and runs inference (requires checkpoint file).
        Optionally generate/update result artifacts.
        """
        clean_key = model_key.lower().replace("-", "_").replace(" ", "_")
        if clean_key not in CLASSICAL_BENCHMARK_CONFIGS:
            raise ValueError(
                f"Model '{model_key}' not recognized. "
                f"Available: {list(CLASSICAL_BENCHMARK_CONFIGS.keys())}"
            )

        cfg = CLASSICAL_BENCHMARK_CONFIGS[clean_key]
        ckpt_path = cfg["checkpoint"]
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Use recompute_from_predictions() for checkpoint-free verification."
            )

        model = joblib.load(ckpt_path)
        X_data, y_true = self.get_split_data(split=split, representation=cfg["representation"])

        print(f"  Running model inference for {cfg['display_name']} on {split} split...")
        t0 = time.time()
        y_pred = model.predict(X_data)
        elapsed = time.time() - t0

        # Canonical 24-class metrics
        m = compute_canonical_metrics(y_true, y_pred, label_space="nasa")

        result_dict = {
            "model_key": clean_key,
            "config_id": cfg["config_id"],
            "display_name": cfg["display_name"],
            "split": split,
            "evaluation_split": split.upper(),
            "num_samples": int(len(y_true)),
            "elapsed_seconds": elapsed,
            "benchmark_acc": cfg["benchmark_acc"],
            "benchmark_macro_f1": cfg["benchmark_macro_f1"],
            **m,
        }

        if generate_artifacts:
            self._save_artifacts(clean_key, split, y_true, y_pred, result_dict)

        return result_dict

    def _save_artifacts(
        self,
        clean_key: str,
        split: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        result_dict: Dict[str, Any],
    ) -> None:
        """Save classification reports, confusion matrices, predictions, and metrics JSON."""
        import json
        from deep_learning.datasets import load_synset_mapping

        cfg = CLASSICAL_BENCHMARK_CONFIGS[clean_key]
        out_dir = _RESULT_DIRS[clean_key]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use canonical NASA ID label space
        # For classical models y_true/y_pred are in NASA IDs
        report_df = generate_classification_report_df(
            y_true, y_pred, label_space="nasa"
        )
        report_df.to_csv(out_dir / f"{split}_classification_report.csv")
        with open(out_dir / f"{split}_classification_report.txt", "w") as f:
            f.write(report_df.to_string())

        # Confusion matrices (24×24, NASA ID order)
        title_base = f"{cfg['display_name']} ({split.capitalize()} Split)"
        plot_and_save_confusion_matrix(
            y_true=y_true, y_pred=y_pred,
            title=f"{title_base} — Confusion Matrix",
            save_path=out_dir / f"{split}_confusion_matrix.png",
            label_space="nasa",
        )
        plot_and_save_normalized_confusion_matrix(
            y_true=y_true, y_pred=y_pred,
            title=f"{title_base} — Normalized Confusion Matrix",
            save_path=out_dir / f"{split}_confusion_matrix_normalized.png",
            label_space="nasa",
        )

        # Prediction arrays
        np.save(out_dir / f"{split}_predictions.npy", y_pred)
        np.save(out_dir / f"{split}_labels.npy", y_true)

        # Split-specific metrics JSON (does NOT overwrite existing final_metrics.json)
        metrics_path = out_dir / f"{split}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"  Generating evaluation metrics → {metrics_path}")

    def evaluate_all(
        self,
        split: str = "test",
        generate_artifacts: bool = False,
    ) -> List[Dict[str, Any]]:
        """Evaluate all 5 classical champion models on the requested split."""
        results = []
        for key in ["knn", "naive_bayes", "decision_tree", "random_forest", "svm"]:
            res = self.evaluate_model(key, split=split, generate_artifacts=generate_artifacts)
            results.append(res)
        return results

    def verify_all_from_predictions(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Checkpoint-free verification of all 5 classical models.
        Loads stored prediction arrays and computes canonical metrics.
        """
        results = []
        for key in ["knn", "naive_bayes", "decision_tree", "random_forest", "svm"]:
            res = self.recompute_from_predictions(key, split=split)
            results.append(res)
        return results


# Global singleton
classical_pipeline = ClassicalPipeline()
