"""
Inference service for the Mars Rover ML API.

Responsibilities:
  1. Load stored test predictions (and probability distributions for DL models).
     - Classical models: results/<dir>/test_predictions.npy
     - Deep learning models: results/<dir>/predictions/test_predictions.npy
                             results/<dir>/predictions/test_probabilities.npy
  2. Lazy-load deep-learning checkpoints on first use, then cache them.
     - Models are NOT pre-loaded at startup.
     - A model is loaded only when selected for live inference.
  3. Live inference for deep learning models via PyTorch.
  4. For classical models, report confidence as null (no calibrated probabilities
     available from sklearn joblib predictions).

Checkpoint policy:
  - Checkpoints (*.pth, *.joblib) are NOT committed to Git.
  - Their local presence is detected at runtime.
  - If absent, classification falls back to stored TEST predictions.
  - For TRAIN/VAL samples, classification requires a live checkpoint.

Inference source labels:
  - "LIVE MODEL"              — actual runtime inference performed
  - "STORED TEST PREDICTION"  — read from precomputed test_predictions.npy
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation.mapping import (
    ACTIVE_INDEX_TO_NASA,
    NASA_CLASS_NAMES,
    NASA_TO_ACTIVE_INDEX,
    to_active_index_array,
)

# ─────────────────────────────────────────────────────────────────────────────
# Inference result dataclass
# ─────────────────────────────────────────────────────────────────────────────

class InferenceResult:
    __slots__ = (
        "predicted_nasa_id",
        "predicted_class_name",
        "confidence",       # float or None
        "top_k",            # list of (nasa_id, class_name, prob) tuples
        "inference_source", # "LIVE MODEL" or "STORED TEST PREDICTION"
        "inference_time_ms", # float or None
    )

    def __init__(
        self,
        predicted_nasa_id: int,
        predicted_class_name: str,
        confidence: Optional[float],
        top_k: List[Tuple[int, str, float]],
        inference_source: str,
        inference_time_ms: Optional[float],
    ):
        self.predicted_nasa_id = predicted_nasa_id
        self.predicted_class_name = predicted_class_name
        self.confidence = confidence
        self.top_k = top_k
        self.inference_source = inference_source
        self.inference_time_ms = inference_time_ms


# ─────────────────────────────────────────────────────────────────────────────
# Stored prediction loader (one shared instance per model, loaded once)
# ─────────────────────────────────────────────────────────────────────────────

class _StoredPredictions:
    """Lazy-loads and caches stored test prediction arrays for one model."""

    def __init__(self, predictions_dir: str, label_space: str):
        self._dir = Path(predictions_dir)
        self._label_space = label_space  # "nasa" or "active_index"
        self._preds: Optional[np.ndarray] = None
        self._probs: Optional[np.ndarray] = None
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        preds_path = self._dir / "test_predictions.npy"
        probs_path = self._dir / "test_probabilities.npy"

        if not preds_path.exists():
            raise FileNotFoundError(f"Stored predictions not found: {preds_path}")

        self._preds = np.load(str(preds_path))
        if probs_path.exists():
            self._probs = np.load(str(probs_path))
        self._loaded = True

    def get(self, test_index: int) -> Tuple[int, Optional[np.ndarray]]:
        """
        Return (predicted_label, prob_vector_or_None) for a test sample.

        predicted_label is a NASA class ID.
        prob_vector is shape (24,) or None for classical models.
        """
        with self._lock:
            if not self._loaded:
                self._load()

        raw_pred = int(self._preds[test_index])

        # Convert active index to NASA ID for DL models
        if self._label_space == "active_index":
            nasa_id = ACTIVE_INDEX_TO_NASA[raw_pred]
        else:
            nasa_id = raw_pred

        prob_vec: Optional[np.ndarray] = None
        if self._probs is not None:
            prob_vec = self._probs[test_index]

        return nasa_id, prob_vec


# ─────────────────────────────────────────────────────────────────────────────
# Deep-learning live-inference model loader (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

class _DLModelLoader:
    """Lazy-loads and caches a single PyTorch deep-learning model."""

    def __init__(self, model_id: str, checkpoint_path: str):
        self._model_id = model_id
        self._checkpoint_path = checkpoint_path
        self._model: Optional[Any] = None
        self._lock = threading.Lock()

    def _build_and_load(self) -> Any:
        import torch

        # Import the appropriate builder function for each model
        if self._model_id == "resnet50":
            from deep_learning.models.resnet50 import build_resnet50_baseline
            model, _ = build_resnet50_baseline(num_classes=24, pretrained=False)
        elif self._model_id == "mrscatt":
            from deep_learning.models.mrscatt import build_mrscatt_model
            model, _ = build_mrscatt_model(num_classes=24, pretrained=False)
        elif self._model_id == "vit":
            from deep_learning.models.vit import build_vit_baseline
            model, _ = build_vit_baseline(num_classes=24, pretrained=False)
        elif self._model_id == "efficientnet_b3":
            from deep_learning.models.efficientnet_b3 import build_efficientnet_b3_baseline
            model, _ = build_efficientnet_b3_baseline(num_classes=24, pretrained=False)
        else:
            raise ValueError(f"Unknown deep learning model id: {self._model_id}")

        ckpt = torch.load(self._checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_model(self) -> Any:
        """Return the loaded (and cached) model — loads on first call."""
        with self._lock:
            if self._model is None:
                self._model = self._build_and_load()
        return self._model

    def run_inference(self, image_path: str) -> Tuple[int, np.ndarray, float]:
        """
        Run live inference on an image.

        Returns: (predicted_nasa_id, prob_vector_shape_24, inference_time_ms)
        """
        import torch
        from PIL import Image
        from deep_learning.transforms import get_eval_transforms

        model = self.get_model()
        transform = get_eval_transforms(target_size=(224, 224))

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

        t_start = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        t_end = time.perf_counter()

        inference_time_ms = (t_end - t_start) * 1000.0
        pred_active_idx = int(np.argmax(probs))
        predicted_nasa_id = ACTIVE_INDEX_TO_NASA[pred_active_idx]

        return predicted_nasa_id, probs, inference_time_ms


# ─────────────────────────────────────────────────────────────────────────────
# Classical model live-inference loader (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

class _ClassicalModelLoader:
    """Lazy-loads and caches a joblib sklearn pipeline for live inference."""

    def __init__(self, model_id: str, checkpoint_path: str):
        self._model_id = model_id
        self._checkpoint_path = checkpoint_path
        self._pipeline = None
        self._lock = threading.Lock()
        self._features_dir: Optional[Path] = None

    def set_features_dir(self, path: Path) -> None:
        self._features_dir = path

    def _load_pipeline(self) -> None:
        import joblib
        self._pipeline = joblib.load(self._checkpoint_path)

    def get_pipeline(self):
        with self._lock:
            if self._pipeline is None:
                self._load_pipeline()
        return self._pipeline

    def run_inference(self, sample_feature_vector: np.ndarray) -> Tuple[int, float]:
        """
        Run live inference for a classical model.

        Returns: (predicted_nasa_id, inference_time_ms)
        Classical models do not produce calibrated probabilities.
        """
        pipeline = self.get_pipeline()
        x = sample_feature_vector.reshape(1, -1).astype(np.float32)

        t_start = time.perf_counter()
        pred = pipeline.predict(x)
        t_end = time.perf_counter()

        inference_time_ms = (t_end - t_start) * 1000.0
        predicted_nasa_id = int(pred[0])
        return predicted_nasa_id, inference_time_ms


# ─────────────────────────────────────────────────────────────────────────────
# InferenceService
# ─────────────────────────────────────────────────────────────────────────────

class InferenceService:
    """
    Routes classification requests to:
    - Stored test predictions (for TEST split samples or when checkpoint is absent).
    - Live model inference (when checkpoint exists and sample is in TEST/VAL/TRAIN).

    For TRAIN or VAL samples, live inference is REQUIRED (no stored predictions).
    """

    def __init__(self, project_root: Path) -> None:
        self._root = project_root

        # Lazy loaders — keyed by model_id
        self._stored_preds: Dict[str, _StoredPredictions] = {}
        self._dl_loaders: Dict[str, _DLModelLoader] = {}
        self._cl_loaders: Dict[str, _ClassicalModelLoader] = {}

        self._features_dir = project_root / "features"

        # Pre-load the test feature matrix reference for classical live inference
        self._test_features: Optional[np.ndarray] = None
        self._features_loaded = False
        self._features_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # Stored prediction cache
    # ─────────────────────────────────────────────────────────────────────────

    def _get_stored_preds(self, model_id: str, predictions_dir: str, label_space: str) -> _StoredPredictions:
        if model_id not in self._stored_preds:
            self._stored_preds[model_id] = _StoredPredictions(predictions_dir, label_space)
        return self._stored_preds[model_id]

    # ─────────────────────────────────────────────────────────────────────────
    # DL model cache
    # ─────────────────────────────────────────────────────────────────────────

    def _get_dl_loader(self, model_id: str, checkpoint_path: str) -> _DLModelLoader:
        if model_id not in self._dl_loaders:
            self._dl_loaders[model_id] = _DLModelLoader(model_id, checkpoint_path)
        return self._dl_loaders[model_id]

    # ─────────────────────────────────────────────────────────────────────────
    # Feature loading for classical live inference
    # ─────────────────────────────────────────────────────────────────────────

    def _get_test_features(self) -> Optional[np.ndarray]:
        """Load the test feature matrix (features/X_test.npz) once and cache it."""
        with self._features_lock:
            if not self._features_loaded:
                feat_path = self._features_dir / "X_test.npz"
                if feat_path.exists():
                    data = np.load(str(feat_path))
                    self._test_features = data["X"]
                self._features_loaded = True
        return self._test_features

    # ─────────────────────────────────────────────────────────────────────────
    # Count of loaded DL checkpoints (for health endpoint)
    # ─────────────────────────────────────────────────────────────────────────

    def count_loaded_checkpoints(self) -> int:
        return sum(1 for loader in self._dl_loaders.values() if loader._model is not None)

    # ─────────────────────────────────────────────────────────────────────────
    # Main classify method
    # ─────────────────────────────────────────────────────────────────────────

    def classify(
        self,
        *,
        model_id: str,
        model_info: dict,
        sample_split: str,
        sample_split_index: int,
        sample_image_path: str,
        prefer_live: bool = False,
    ) -> InferenceResult:
        """
        Perform classification and return an InferenceResult.

        Decision tree:
          1. If sample is in TEST split AND (checkpoint absent OR prefer_live=False):
             → Use stored test prediction.
          2. If sample is in TEST split AND checkpoint present AND prefer_live=True:
             → Use live model inference.
          3. If sample is in TRAIN or VAL split:
             → REQUIRE live inference (stored predictions only cover TEST).
             → If checkpoint absent, raise an error.
        """
        is_test = sample_split == "test"
        has_checkpoint = model_info["has_checkpoint"]
        avail_checkpoint = model_info.get("best_checkpoint")  # may be None
        is_dl = model_info.get("is_deep_learning", False)
        label_space = model_info["label_space"]
        predictions_dir = model_info.get("predictions_dir", "")

        # ── Case 1/2: TEST sample ────────────────────────────────────────────
        if is_test:
            if has_checkpoint and prefer_live and avail_checkpoint:
                # Live inference requested and checkpoint exists
                return self._live_inference(
                    model_id=model_id,
                    is_dl=is_dl,
                    checkpoint_path=avail_checkpoint,
                    image_path=sample_image_path,
                    label_space=label_space,
                )
            else:
                # Default: use stored test prediction
                return self._stored_inference(
                    model_id=model_id,
                    predictions_dir=predictions_dir,
                    label_space=label_space,
                    test_index=sample_split_index,
                )

        # ── Case 3: TRAIN or VAL sample ─────────────────────────────────────
        if not has_checkpoint or not avail_checkpoint:
            raise RuntimeError(
                f"No checkpoint available for model '{model_id}'. "
                f"Live inference is required for {sample_split.upper()} samples "
                f"because stored predictions only cover the TEST split."
            )

        return self._live_inference(
            model_id=model_id,
            is_dl=is_dl,
            checkpoint_path=avail_checkpoint,
            image_path=sample_image_path,
            label_space=label_space,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stored prediction path
    # ─────────────────────────────────────────────────────────────────────────

    def _stored_inference(
        self,
        model_id: str,
        predictions_dir: str,
        label_space: str,
        test_index: int,
    ) -> InferenceResult:
        sp = self._get_stored_preds(model_id, predictions_dir, label_space)
        predicted_nasa_id, prob_vec = sp.get(test_index)
        class_name = NASA_CLASS_NAMES.get(predicted_nasa_id, f"class_{predicted_nasa_id}")

        top_k: List[Tuple[int, str, float]] = []
        confidence: Optional[float] = None

        if prob_vec is not None:
            # DL model: extract top-5 from probability distribution
            top_indices = np.argsort(prob_vec)[::-1][:5]
            confidence = float(prob_vec[int(np.argmax(prob_vec))])
            for rank_idx in top_indices:
                nasa_id = ACTIVE_INDEX_TO_NASA[int(rank_idx)]
                top_k.append((
                    nasa_id,
                    NASA_CLASS_NAMES.get(nasa_id, f"class_{nasa_id}"),
                    float(prob_vec[rank_idx]),
                ))

        return InferenceResult(
            predicted_nasa_id=predicted_nasa_id,
            predicted_class_name=class_name,
            confidence=confidence,
            top_k=top_k,
            inference_source="STORED TEST PREDICTION",
            inference_time_ms=None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Live inference path
    # ─────────────────────────────────────────────────────────────────────────

    def _live_inference(
        self,
        model_id: str,
        is_dl: bool,
        checkpoint_path: str,
        image_path: str,
        label_space: str,
    ) -> InferenceResult:
        if is_dl:
            return self._live_dl_inference(model_id, checkpoint_path, image_path)
        else:
            return self._live_classical_inference(model_id, checkpoint_path, image_path)

    def _live_dl_inference(
        self,
        model_id: str,
        checkpoint_path: str,
        image_path: str,
    ) -> InferenceResult:
        loader = self._get_dl_loader(model_id, checkpoint_path)
        predicted_nasa_id, prob_vec, inference_time_ms = loader.run_inference(image_path)
        class_name = NASA_CLASS_NAMES.get(predicted_nasa_id, f"class_{predicted_nasa_id}")

        confidence = float(prob_vec[int(np.argmax(prob_vec))])
        top_indices = np.argsort(prob_vec)[::-1][:5]
        top_k = [
            (ACTIVE_INDEX_TO_NASA[int(i)],
             NASA_CLASS_NAMES.get(ACTIVE_INDEX_TO_NASA[int(i)], f"class_{i}"),
             float(prob_vec[i]))
            for i in top_indices
        ]

        return InferenceResult(
            predicted_nasa_id=predicted_nasa_id,
            predicted_class_name=class_name,
            confidence=confidence,
            top_k=top_k,
            inference_source="LIVE MODEL",
            inference_time_ms=round(inference_time_ms, 2),
        )

    def _live_classical_inference(
        self,
        model_id: str,
        checkpoint_path: str,
        image_path: str,
    ) -> InferenceResult:
        """
        Classical models require the pre-extracted feature vector.
        We load features from the features/X_test.npz when available,
        or extract on-the-fly from the image.
        """
        import joblib

        pipeline = joblib.load(checkpoint_path)

        # Try to load pre-extracted features for this image
        # The test features matrix is aligned with test-calibrated-shuffled.txt
        # For TRAIN/VAL live inference we need on-the-fly feature extraction
        try:
            from preprocessing.image_processing import preprocess_image
            from preprocessing.feature_extraction import extract_features_from_preprocessed_image

            t_start = time.perf_counter()
            preprocessed = preprocess_image(image_path, target_size=(256, 256))
            feat_vec = extract_features_from_preprocessed_image(preprocessed)
            x = feat_vec.reshape(1, -1).astype(np.float32)
            pred = pipeline.predict(x)
            t_end = time.perf_counter()

            predicted_nasa_id = int(pred[0])
            inference_time_ms = (t_end - t_start) * 1000.0

        except Exception as e:
            raise RuntimeError(
                f"Classical live inference failed for model '{model_id}': {e}"
            )

        class_name = NASA_CLASS_NAMES.get(predicted_nasa_id, f"class_{predicted_nasa_id}")

        return InferenceResult(
            predicted_nasa_id=predicted_nasa_id,
            predicted_class_name=class_name,
            confidence=None,  # Classical models: no calibrated probabilities
            top_k=[],
            inference_source="LIVE MODEL",
            inference_time_ms=round(inference_time_ms, 2),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        project_root = Path(__file__).resolve().parent.parent
        _inference_service = InferenceService(project_root)
    return _inference_service
