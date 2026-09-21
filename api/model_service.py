"""
Model service for the Mars Rover ML API.

Provides:
  - Canonical metadata for all 9 models (no fabricated specs).
  - Benchmark metrics loaded from the project's canonical
    results/10_Overall_Comparison/final_test_comparison.json.
  - Checkpoint availability detection (saved_models/).
  - Stored-prediction file availability detection (results/).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Static model registry (factual descriptions only, no invented architectures)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: List[dict] = [
    # ── Classical ML (5 models) ──────────────────────────────────────────────
    {
        "id": "knn",
        "name": "K-Nearest Neighbors",
        "display_name": "K-Nearest Neighbors",
        "category": "classical_ml",
        "architecture": "K-Nearest Neighbors",
        "input_representation": "Scaled + PCA-100 (8,186 features)",
        "configuration": "K = 15, Distance weighting, Scaled + PCA-100",
        "description": (
            "K-Nearest Neighbors classifier operating on normalized HOG, LBP, and color histogram "
            "features projected via PCA. Distance-weighted variant."
        ),
        "checkpoint_paths": ["saved_models/knn/KNN-1.joblib", "saved_models/knn/KNN-4.joblib"],
        "predictions_dir": "results/01_KNN",
        "label_space": "nasa",
        "is_deep_learning": False,
    },
    {
        "id": "naive_bayes",
        "name": "Gaussian Naive Bayes",
        "display_name": "Gaussian Naive Bayes",
        "category": "classical_ml",
        "architecture": "Gaussian Naive Bayes",
        "input_representation": "8,186 engineered features",
        "configuration": "Scaled features, Uniform priors",
        "description": (
            "Gaussian Naive Bayes classifier with StandardScaler preprocessing on raw HOG, "
            "LBP, and color histogram feature concatenation."
        ),
        "checkpoint_paths": ["saved_models/naive_bayes/NB-1.joblib", "saved_models/naive_bayes/NB-2.joblib"],
        "predictions_dir": "results/02_Naive_Bayes",
        "label_space": "nasa",
        "is_deep_learning": False,
    },
    {
        "id": "decision_tree",
        "name": "Decision Tree",
        "display_name": "Decision Tree",
        "category": "classical_ml",
        "architecture": "Decision Tree",
        "input_representation": "8,186 engineered features",
        "configuration": "Max depth = 15, Balanced class weighting",
        "description": (
            "Decision Tree classifier with balanced class weights, "
            "trained on the same HOG+LBP+color-histogram feature set."
        ),
        "checkpoint_paths": ["saved_models/decision_tree/DT-1.joblib", "saved_models/decision_tree/DT-2.joblib"],
        "predictions_dir": "results/03_Decision_Tree",
        "label_space": "nasa",
        "is_deep_learning": False,
    },
    {
        "id": "random_forest",
        "name": "Random Forest",
        "display_name": "Random Forest",
        "category": "classical_ml",
        "architecture": "Random Forest",
        "input_representation": "8,186 engineered features",
        "configuration": "100 trees, Max depth = 15, Balanced class weighting",
        "description": (
            "Ensemble of 200 bootstrap decision trees with balanced-subsample class weighting "
            "and min_samples_leaf regularization."
        ),
        "checkpoint_paths": ["saved_models/random_forest/RF-1.joblib", "saved_models/random_forest/RF-2.joblib"],
        "predictions_dir": "results/04_Random_Forest",
        "label_space": "nasa",
        "is_deep_learning": False,
    },
    {
        "id": "svm",
        "name": "Support Vector Machine",
        "display_name": "Support Vector Machine",
        "category": "classical_ml",
        "architecture": "Support Vector Machine",
        "input_representation": "Scaled + PCA-100 (8,186 features)",
        "configuration": "RBF kernel, C = 1, Gamma = scale, Balanced class weighting, Scaled + PCA-100",
        "description": (
            "RBF-kernel Support Vector Classifier with StandardScaler and PCA (100 components) "
            "preprocessing, trained on engineered features with balanced class weights."
        ),
        "checkpoint_paths": ["saved_models/svm/SVM-1.joblib", "saved_models/svm/SVM-4.joblib"],
        "predictions_dir": "results/05_SVM",
        "label_space": "nasa",
        "is_deep_learning": False,
    },
    # ── Deep Learning (4 models) ──────────────────────────────────────────────
    {
        "id": "mrscatt",
        "name": "MRSCAtt",
        "display_name": "MRSCAtt",
        "category": "deep_learning",
        "architecture": "Multi-Resolution Spatial-Channel Attention",
        "input_representation": "224 × 224 RGB image",
        "configuration": "ResNet-50 layer4 backbone, Channel attention, Spatial attention, Global average pooling, 24-class classifier",
        "description": (
            "Multi-Resolution Spatial-Channel Attention network built on a ResNet-50 backbone "
            "with channel and spatial attention modules specialized for planetary surface imagery."
        ),
        "checkpoint_paths": [
            "saved_models/deep_learning/mrscatt/best_mrscatt.pth",
            "saved_models/deep_learning/mrscatt/last_mrscatt.pth",
        ],
        "predictions_dir": "results/06_MRSCAtt/predictions",
        "label_space": "active_index",
        "is_deep_learning": True,
    },
    {
        "id": "vit",
        "name": "ViT-B/16",
        "display_name": "ViT-B/16",
        "category": "deep_learning",
        "architecture": "Vision Transformer B/16",
        "input_representation": "224 × 224 RGB image",
        "configuration": "Vision Transformer, 16×16 patches, 224×224 input, 24 classes",
        "description": (
            "Vision Transformer with 16×16 patch resolution and full fine-tuning "
            "on the 24-class Mars surface taxonomy."
        ),
        "checkpoint_paths": [
            "saved_models/deep_learning/vit/best_vit.pth",
            "saved_models/deep_learning/vit/last_vit.pth",
        ],
        "predictions_dir": "results/07_ViT_B16/predictions",
        "label_space": "active_index",
        "is_deep_learning": True,
    },
    {
        "id": "resnet50",
        "name": "ResNet-50",
        "display_name": "ResNet-50",
        "category": "deep_learning",
        "architecture": "Residual Network 50",
        "input_representation": "224 × 224 RGB image",
        "configuration": "Pretrained ResNet-50, 224×224 input, 24-class classifier, Dropout 0.2",
        "description": (
            "50-layer skip-connection convolutional network fine-tuned from ImageNet weights "
            "with a 24-class classification head and dropout regularization."
        ),
        "checkpoint_paths": [
            "saved_models/deep_learning/resnet50/best_resnet50.pth",
            "saved_models/deep_learning/resnet50/last_resnet50.pth",
        ],
        "predictions_dir": "results/08_ResNet50/predictions",
        "label_space": "active_index",
        "is_deep_learning": True,
    },
    {
        "id": "efficientnet_b3",
        "name": "EfficientNet-B3",
        "display_name": "EfficientNet-B3",
        "category": "deep_learning",
        "architecture": "EfficientNet B3",
        "input_representation": "300 × 300 RGB image",
        "configuration": "Pretrained EfficientNet-B3, 300×300 input, 24-class classifier, Dropout 0.3",
        "description": (
            "Compound-scaled EfficientNet B3 fine-tuned from ImageNet-21k weights "
            "with progressive-resolution training on the 24-class Mars surface dataset."
        ),
        "checkpoint_paths": [
            "saved_models/deep_learning/efficientnet_b3/best_efficientnet_b3.pth",
            "saved_models/deep_learning/efficientnet_b3/last_efficientnet_b3.pth",
        ],
        "predictions_dir": "results/09_EfficientNet_B3/predictions",
        "label_space": "active_index",
        "is_deep_learning": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# ModelService
# ─────────────────────────────────────────────────────────────────────────────

class ModelService:
    """Provides model metadata and benchmark metrics loaded from canonical files."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._performance_path = (
            project_root / "results" / "10_Overall_Comparison" / "final_test_comparison.json"
        )
        self._performance_cache: Optional[dict] = None
        self._availability_cache: Optional[Dict[str, dict]] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _check_checkpoint_available(self, checkpoint_paths: List[str]) -> bool:
        """Return True if any candidate checkpoint path exists on disk."""
        for cp in checkpoint_paths:
            if os.path.isfile(self._root / cp):
                return True
        return False

    def _check_stored_predictions_available(self, predictions_dir: str) -> bool:
        """Return True if test_predictions.npy exists in the given directory."""
        pred_path = self._root / predictions_dir / "test_predictions.npy"
        return os.path.isfile(pred_path)

    def _best_checkpoint_path(self, checkpoint_paths: List[str]) -> Optional[str]:
        """Return absolute path to the first available checkpoint."""
        for cp in checkpoint_paths:
            full = self._root / cp
            if os.path.isfile(full):
                return str(full)
        return None

    def _load_performance(self) -> dict:
        if self._performance_cache is None:
            with open(self._performance_path, "r", encoding="utf-8") as fh:
                self._performance_cache = json.load(fh)
        return self._performance_cache

    def _build_availability_map(self) -> Dict[str, dict]:
        """Build per-model availability dict once and cache it."""
        if self._availability_cache is not None:
            return self._availability_cache

        avail: Dict[str, dict] = {}
        for defn in _MODEL_REGISTRY:
            mid = defn["id"]
            has_ckpt = self._check_checkpoint_available(defn["checkpoint_paths"])
            has_preds = self._check_stored_predictions_available(defn["predictions_dir"])
            avail[mid] = {
                "has_checkpoint": has_ckpt,
                "has_stored_predictions": has_preds,
                "available": has_ckpt or has_preds,
                "best_checkpoint": self._best_checkpoint_path(defn["checkpoint_paths"])
                if has_ckpt else None,
                "predictions_dir": str(self._root / defn["predictions_dir"]),
            }

        self._availability_cache = avail
        return avail

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get_models(self) -> List[dict]:
        """Return list of all model info dicts with live availability status."""
        avail = self._build_availability_map()
        results = []
        for defn in _MODEL_REGISTRY:
            mid = defn["id"]
            a = avail[mid]
            results.append({
                "id": mid,
                "name": defn["name"],
                "display_name": defn["display_name"],
                "category": defn["category"],
                "architecture": defn["architecture"],
                "input_representation": defn.get("input_representation"),
                "configuration": defn.get("configuration"),
                "description": defn["description"],
                "available": a["available"],
                "has_checkpoint": a["has_checkpoint"],
                "has_stored_predictions": a["has_stored_predictions"],
                "label_space": defn["label_space"],
                "is_deep_learning": defn["is_deep_learning"],
            })
        return results

    def get_model_by_id(self, model_id: str) -> Optional[dict]:
        """Return a single model's full info dict or None."""
        models = {m["id"]: m for m in self.get_models()}
        return models.get(model_id)

    def get_performance(self) -> dict:
        """Load and return the canonical TEST benchmark from JSON."""
        return self._load_performance()

    def get_model_performance_by_id(self, model_id: str) -> Optional[dict]:
        """Return the benchmark dict for a specific model key."""
        perf = self._load_performance()
        for m in perf.get("models", []):
            if m["key"] == model_id:
                return m
        return None

    def get_availability(self, model_id: str) -> Optional[dict]:
        """Return availability dict for a model."""
        avail = self._build_availability_map()
        return avail.get(model_id)

    def get_registry_entry(self, model_id: str) -> Optional[dict]:
        """Return the raw static registry entry for a model."""
        for defn in _MODEL_REGISTRY:
            if defn["id"] == model_id:
                return defn
        return None

    def count_available(self) -> int:
        avail = self._build_availability_map()
        return sum(1 for a in avail.values() if a["available"])


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        project_root = Path(__file__).resolve().parent.parent
        _model_service = ModelService(project_root)
    return _model_service
