"""
Pydantic schemas for Mars Rover ML Classification API.

All response types strictly reflect real project data.
No fabricated metrics or invented model outputs are used.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    """Metadata for a single classification model."""
    id: str = Field(..., description="Unique model key, e.g. 'knn', 'resnet50'")
    name: str = Field(..., description="Human-readable display name")
    category: str = Field(..., description="'classical_ml' or 'deep_learning'")
    architecture: str = Field(..., description="Architecture family")
    input_representation: Optional[str] = Field(default=None, description="Input representation, resolution, or features")
    configuration: Optional[str] = Field(default=None, description="Verified model configuration")
    description: str = Field(..., description="Brief factual description")
    available: bool = Field(..., description="Whether this model can perform classification")
    has_checkpoint: bool = Field(..., description="Whether a local .pth / .joblib checkpoint is present")
    has_stored_predictions: bool = Field(..., description="Whether stored test predictions exist")
    label_space: str = Field(..., description="'nasa' for classical, 'active_index' for DL")


class ModelBenchmark(BaseModel):
    """Canonical TEST-split benchmark metrics from final_test_comparison.json."""
    rank: int
    key: str
    model: str
    category: str
    n_test_samples: int
    accuracy_pct: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    weighted_f1: float
    label_space: str


class ModelPerformanceResponse(BaseModel):
    evaluation_split: str
    metric_protocol: str
    num_active_classes: int
    num_test_samples: int
    models: List[ModelBenchmark]


# ─────────────────────────────────────────────────────────────────────────────
# Samples
# ─────────────────────────────────────────────────────────────────────────────

class SampleItem(BaseModel):
    """A single dataset sample with metadata."""
    sample_id: str = Field(..., description="Unique ID, e.g. 'TEST-0042'")
    filename: str = Field(..., description="Bare filename, e.g. '0830MR0036510000500684E01_DRCL.JPG'")
    split: str = Field(..., description="'train', 'val', or 'test'")
    split_index: int = Field(..., description="Zero-based index within its split")
    ground_truth_id: int = Field(..., description="NASA class ID (0–24, skipping 22)")
    ground_truth_name: str = Field(..., description="Human-readable class label")
    image_url: str = Field(..., description="URL to stream the image via /api/samples/{id}/image")
    has_stored_prediction: bool = Field(..., description="True when the sample is in the TEST split")


class SampleListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    samples: List[SampleItem]


class SampleDetailResponse(SampleItem):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

class TopPrediction(BaseModel):
    rank: int
    class_id: int          # NASA ID
    class_name: str
    probability: float     # 0.0 – 1.0


class ClassifyRequest(BaseModel):
    sample_id: str = Field(..., description="Sample ID from /api/samples")
    model: str = Field(..., description="Model key, e.g. 'resnet50'")
    prefer_live: bool = Field(
        default=False,
        description="Prefer live model inference over stored test predictions when both are available",
    )


class ClassifyResponse(BaseModel):
    sample_id: str
    model: str
    model_name: str

    predicted_class_id: int           # NASA ID
    predicted_class_name: str
    ground_truth_id: int              # NASA ID
    ground_truth_name: str
    correct: bool

    # Confidence: real softmax prob for DL models; null for classical models
    confidence: Optional[float] = None
    confidence_display: str           # e.g. "98.45%" or "NOT AVAILABLE"

    # Top-K only populated for DL models (has probability distributions)
    top_predictions: List[TopPrediction] = []

    # Source of the inference result
    inference_source: str             # "LIVE MODEL" or "STORED TEST PREDICTION"
    # Measured latency for LIVE MODEL; null for STORED TEST PREDICTION
    inference_time_ms: Optional[float] = None
    inference_time_display: str       # "12.4 ms" or "N/A — STORED PREDICTION"

    status: str                       # "success", "stored_prediction", "checkpoint_unavailable"

    # Denormalized model-level test metrics for the result panel
    model_test_metrics: Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool
    test_samples: int
    val_samples: int
    train_samples: int
    models_available: int
    checkpoints_loaded: int
