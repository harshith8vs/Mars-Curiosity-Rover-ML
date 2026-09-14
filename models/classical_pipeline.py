"""Classical machine learning pipeline consolidating training, evaluation, and inference."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from evaluation.metrics import compute_metrics, generate_classification_report_df
from models.decision_tree import build_decision_tree_model, train_decision_tree
from models.imbalance import compute_balanced_class_weights
from models.knn import build_knn_model, train_knn
from models.naive_bayes import build_naive_bayes_model, train_naive_bayes
from models.random_forest import build_random_forest_model, train_random_forest
from models.scaling import TrainingScaler
from models.svm import build_svm_model, train_svm


# Mapping of champion classical configurations
CLASSICAL_BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "knn": {
        "config_id": "KNN-4",
        "display_name": "KNN (Scaled+PCA, Distance)",
        "representation": "pca",
        "checkpoint": config.SAVED_MODELS_DIR / "knn" / "KNN-4.joblib",
        "benchmark_acc": 50.85,
        "benchmark_macro_f1": 0.6613,
    },
    "naive_bayes": {
        "config_id": "NB-2",
        "display_name": "GaussianNB (Scaled, Uniform Priors)",
        "representation": "scaled",
        "checkpoint": config.SAVED_MODELS_DIR / "naive_bayes" / "NB-2.joblib",
        "benchmark_acc": 12.07,
        "benchmark_macro_f1": 0.2012,
    },
    "decision_tree": {
        "config_id": "DT-2",
        "display_name": "DecisionTree (Unscaled, Balanced)",
        "representation": "raw",
        "checkpoint": config.SAVED_MODELS_DIR / "decision_tree" / "DT-2.joblib",
        "benchmark_acc": 15.30,
        "benchmark_macro_f1": 0.2690,
    },
    "random_forest": {
        "config_id": "RF-2",
        "display_name": "RandomForest (Unscaled, Balanced)",
        "representation": "raw",
        "checkpoint": config.SAVED_MODELS_DIR / "random_forest" / "RF-2.joblib",
        "benchmark_acc": 66.04,
        "benchmark_macro_f1": 0.5993,
    },
    "svm": {
        "config_id": "SVM-4",
        "display_name": "SVM (Scaled+PCA, Balanced)",
        "representation": "pca",
        "checkpoint": config.SAVED_MODELS_DIR / "svm" / "SVM-4.joblib",
        "benchmark_acc": 67.01,
        "benchmark_macro_f1": 0.6393,
    },
}


class ClassicalPipeline:
    """Manages data loading, preprocessors (scaler, PCA), evaluation, and training for classical models."""

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
        """Loads cached feature matrices and labels from features/ directory."""
        if self._X_train is None:
            self._X_train = np.load(config.FEATURES_DIR / "X_train.npz")["X"]
            self._y_train = np.load(config.FEATURES_DIR / "y_train.npy")
            self._X_val = np.load(config.FEATURES_DIR / "X_val.npz")["X"]
            self._y_val = np.load(config.FEATURES_DIR / "y_val.npy")
            self._X_test = np.load(config.FEATURES_DIR / "X_test.npz")["X"]
            self._y_test = np.load(config.FEATURES_DIR / "y_test.npy")

    def get_preprocessors(self) -> Tuple[TrainingScaler, PCA]:
        """Fits TrainingScaler and PCA(n_components=100) exclusively on training data."""
        self.load_features()
        if self._scaler is None:
            self._scaler = TrainingScaler()
            self._scaler.fit(self._X_train)
        if self._pca is None:
            X_train_scaled = self._scaler.transform(self._X_train)
            self._pca = PCA(n_components=100, random_state=42)
            self._pca.fit(X_train_scaled)
        return self._scaler, self._pca

    def get_split_data(self, split: str = "val", representation: str = "raw") -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve features and labels for requested split ('train', 'val', 'test') and representation ('raw', 'scaled', 'pca')."""
        self.load_features()
        scaler, pca = self.get_preprocessors()

        if split == "train":
            X, y = self._X_train, self._y_train
        elif split == "val":
            X, y = self._X_val, self._y_val
        elif split == "test":
            X, y = self._X_test, self._y_test
        else:
            raise ValueError(f"Unknown split: {split}. Choose 'train', 'val', or 'test'.")

        if representation == "raw":
            return X, y
        elif representation == "scaled":
            return scaler.transform(X), y
        elif representation == "pca":
            return pca.transform(scaler.transform(X)), y
        else:
            raise ValueError(f"Unknown representation: {representation}. Choose 'raw', 'scaled', or 'pca'.")

    def evaluate_model(
        self,
        model_key: str,
        split: str = "val",
        generate_artifacts: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a trained classical model on the requested split and optionally generate artifacts."""
        clean_key = model_key.lower().replace("-", "_").replace(" ", "_")
        if clean_key not in CLASSICAL_BENCHMARK_CONFIGS:
            raise ValueError(f"Model {model_key} not recognized. Available: {list(CLASSICAL_BENCHMARK_CONFIGS.keys())}")

        cfg = CLASSICAL_BENCHMARK_CONFIGS[clean_key]
        ckpt_path = cfg["checkpoint"]
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = joblib.load(ckpt_path)
        X_data, y_true = self.get_split_data(split=split, representation=cfg["representation"])

        t0 = time.time()
        y_pred = model.predict(X_data)
        elapsed = time.time() - t0

        acc = float(accuracy_score(y_true, y_pred) * 100.0)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        result_dict = {
            "model_key": clean_key,
            "config_id": cfg["config_id"],
            "display_name": cfg["display_name"],
            "split": split,
            "num_samples": len(y_true),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "weighted_f1": weighted_f1,
            "elapsed_seconds": elapsed,
            "benchmark_acc": cfg["benchmark_acc"],
            "benchmark_macro_f1": cfg["benchmark_macro_f1"],
        }

        if generate_artifacts:
            from deep_learning.datasets import load_synset_mapping
            from evaluation.metrics import (
                generate_classification_report_df,
                plot_and_save_confusion_matrix,
                plot_and_save_normalized_confusion_matrix
            )
            import json

            dir_mapping = {
                "knn": config.RESULTS_KNN_DIR,
                "naive_bayes": config.RESULTS_NAIVE_BAYES_DIR,
                "decision_tree": config.RESULTS_DECISION_TREE_DIR,
                "random_forest": config.RESULTS_RANDOM_FOREST_DIR,
                "svm": config.RESULTS_SVM_DIR,
            }
            out_dir = dir_mapping[clean_key]
            out_dir.mkdir(parents=True, exist_ok=True)

            class_names = load_synset_mapping(config.CLASS_MAPPING_PATH)

            # Classification report CSV and TXT
            report_df = generate_classification_report_df(y_true, y_pred, class_names=class_names, total_classes=25)
            report_df.to_csv(out_dir / f"{split}_classification_report.csv")
            with open(out_dir / f"{split}_classification_report.txt", "w") as f:
                f.write(report_df.to_string())

            # Confusion matrices (raw counts and normalized)
            plot_and_save_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                title=f"{cfg['display_name']} — Confusion Matrix ({split.capitalize()} Split)",
                save_path=out_dir / f"{split}_confusion_matrix.png",
                total_classes=25
            )
            plot_and_save_normalized_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                title=f"{cfg['display_name']} — Normalized Confusion Matrix ({split.capitalize()} Split)",
                save_path=out_dir / f"{split}_confusion_matrix_normalized.png",
                total_classes=25
            )

            # Prediction arrays
            np.save(out_dir / f"{split}_predictions.npy", y_pred)
            np.save(out_dir / f"{split}_labels.npy", y_true)

            # Final metrics JSON
            with open(out_dir / "final_metrics.json", "w") as f:
                json.dump(result_dict, f, indent=2)

        return result_dict

    def evaluate_all(self, split: str = "val", generate_artifacts: bool = False) -> List[Dict[str, Any]]:
        """Evaluates all 5 classical champion models on requested split."""
        results = []
        for key in ["knn", "naive_bayes", "decision_tree", "random_forest", "svm"]:
            res = self.evaluate_model(key, split=split, generate_artifacts=generate_artifacts)
            results.append(res)
        return results


# Global singleton instance for easy import
classical_pipeline = ClassicalPipeline()

