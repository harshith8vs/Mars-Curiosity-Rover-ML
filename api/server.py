"""
FastAPI server for the Mars Rover ML Classification API.

All endpoints serve real data from the project repository.
No fabricated predictions, metrics, or telemetry are returned.

Start with:
    cd "Mars ML models"
    ./venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

CORS is configured to allow:
    http://localhost:5173   (Vite dev server)
    http://127.0.0.1:5173
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api.dataset_service import DatasetService, get_dataset_service
from api.inference_service import get_inference_service
from api.model_service import ModelService, get_model_service
from api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    ModelBenchmark,
    ModelInfo,
    ModelPerformanceResponse,
    SampleDetailResponse,
    SampleItem,
    SampleListResponse,
    TopPrediction,
)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mars Rover ML Classification API",
    description=(
        "Real-data API for the NASA Curiosity Rover ML classification system. "
        "Returns only verified project data; no fabricated metrics or predictions."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Allow Vite dev server and localhost variants
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ds() -> DatasetService:
    return get_dataset_service()


def _ms() -> ModelService:
    return get_model_service()


def _make_image_url(request: Request, sample_id: str) -> str:
    """Build the image streaming URL for a given sample_id."""
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/samples/{sample_id}/image"


def _sample_to_schema(rec, request: Request) -> SampleItem:
    return SampleItem(
        sample_id=rec.sample_id,
        filename=rec.filename,
        split=rec.split,
        split_index=rec.split_index,
        ground_truth_id=rec.ground_truth_id,
        ground_truth_name=rec.ground_truth_name,
        image_url=_make_image_url(request, rec.sample_id),
        has_stored_prediction=rec.split == "test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health(request: Request):
    """System health check. Returns dataset counts and model availability."""
    ds = _ds()
    ms = _ms()
    infer = get_inference_service()
    counts = ds.counts()
    return HealthResponse(
        status="ok",
        dataset_loaded=counts["total"] > 0,
        test_samples=counts["test"],
        val_samples=counts["val"],
        train_samples=counts["train"],
        models_available=ms.count_available(),
        checkpoints_loaded=infer.count_loaded_checkpoints(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/models", response_model=list[ModelInfo], tags=["Models"])
def list_models():
    """
    Return metadata for all 9 models.

    Availability fields reflect actual checkpoint presence on disk.
    """
    ms = _ms()
    raw = ms.get_models()
    return [
        ModelInfo(
            id=m["id"],
            name=m["display_name"],
            category=m["category"],
            architecture=m["architecture"],
            input_representation=m.get("input_representation"),
            configuration=m.get("configuration"),
            description=m["description"],
            available=m["available"],
            has_checkpoint=m["has_checkpoint"],
            has_stored_predictions=m["has_stored_predictions"],
            label_space=m["label_space"],
        )
        for m in raw
    ]


@app.get("/api/benchmarks", response_model=ModelPerformanceResponse, tags=["Models"])
@app.get("/api/models/performance", response_model=ModelPerformanceResponse, tags=["Models"])
def model_performance():
    """
    Return canonical TEST-split benchmark metrics for all 9 models.

    Loaded directly from results/10_Overall_Comparison/final_test_comparison.json.
    Values are never hardcoded in the API layer.
    """
    ms = _ms()
    perf = ms.get_performance()
    benchmarks = [
        ModelBenchmark(
            rank=m["rank"],
            key=m["key"],
            model=m["model"],
            category=m["category"],
            n_test_samples=m["n_test_samples"],
            accuracy_pct=round(m["accuracy_pct"], 4),
            macro_f1=round(m["macro_f1"], 6),
            macro_precision=round(m["macro_precision"], 6),
            macro_recall=round(m["macro_recall"], 6),
            weighted_f1=round(m["weighted_f1"], 6),
            label_space=m["label_space"],
        )
        for m in perf["models"]
    ]
    return ModelPerformanceResponse(
        evaluation_split=perf["evaluation_split"],
        metric_protocol=perf["metric_protocol"],
        num_active_classes=perf["num_active_classes"],
        num_test_samples=perf["num_test_samples"],
        models=benchmarks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Samples
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/samples", response_model=SampleListResponse, tags=["Dataset"])
def list_samples(
    request: Request,
    split: Optional[str] = Query(default=None, description="Filter: 'test', 'val', or 'train'"),
    class_id: Optional[int] = Query(default=None, description="Filter by NASA class ID"),
    q: Optional[str] = Query(default=None, description="Keyword search on filename / class name"),
    page: int = Query(default=1, ge=1, description="1-indexed page number"),
    page_size: int = Query(default=24, ge=1, le=100, description="Items per page"),
):
    """
    List real dataset samples with pagination, split filtering, class filtering, and search.

    Returns TEST samples by default (preferred for classification workflow).
    """
    ds = _ds()
    records, total = ds.list_samples(
        split=split,
        class_id=class_id,
        query=q,
        page=page,
        page_size=page_size,
    )
    total_pages = max(1, math.ceil(total / page_size))
    return SampleListResponse(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        samples=[_sample_to_schema(r, request) for r in records],
    )


@app.get("/api/samples/{sample_id}", response_model=SampleDetailResponse, tags=["Dataset"])
def get_sample(sample_id: str, request: Request):
    """Return metadata for a specific sample."""
    ds = _ds()
    rec = ds.get_sample(sample_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
    return _sample_to_schema(rec, request)


@app.get("/api/samples/{sample_id}/image", tags=["Dataset"])
def serve_sample_image(sample_id: str):
    """
    Stream the actual Mars rover image file.

    Safe: path is resolved via the DatasetService which validates it stays
    within the 'Mars rover data/calibrated/' directory.
    """
    ds = _ds()
    img_path = ds.get_image_path(sample_id)
    if img_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Image for sample '{sample_id}' not found on disk.",
        )
    return FileResponse(
        path=img_path,
        media_type="image/jpeg",
        filename=os.path.basename(img_path),
    )


@app.get("/api/dataset/stats", tags=["Dataset"])
def dataset_stats():
    """Return dataset distribution and breakdown per class and split."""
    ds = _ds()
    return ds.get_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Classify
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/classify", response_model=ClassifyResponse, tags=["Classification"])
def classify(body: ClassifyRequest, request: Request):
    """
    Classify a Mars rover image using the selected model.

    - TEST samples: defaults to stored test predictions (real benchmark results).
      If prefer_live=True and checkpoint exists, runs live model inference.
    - TRAIN/VAL samples: requires a local checkpoint for live inference.
      Returns 422 if no checkpoint is available.

    Confidence is only populated for deep-learning models with probability distributions.
    Classical models always return confidence=null.
    Inference time is only measured for LIVE MODEL runs; null for stored predictions.
    """
    ds = _ds()
    ms = _ms()
    infer = get_inference_service()

    # Validate sample
    rec = ds.get_sample(body.sample_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Sample '{body.sample_id}' not found.")

    # Validate model
    model_info_raw = ms.get_models()
    model_dict = {m["id"]: m for m in model_info_raw}
    if body.model not in model_dict:
        raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found.")

    m_info = model_dict[body.model]
    avail = ms.get_availability(body.model)

    # Enrich model_info with availability details
    m_info["has_checkpoint"] = avail["has_checkpoint"]
    m_info["best_checkpoint"] = avail["best_checkpoint"]
    m_info["predictions_dir"] = avail["predictions_dir"]

    # Check if this non-test sample requires a checkpoint
    if rec.split != "test" and not avail["has_checkpoint"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{body.model}' has no local checkpoint. "
                f"Live inference is required for {rec.split.upper()} samples "
                f"(stored predictions only cover the TEST split). "
                f"Please select a TEST sample or add the model checkpoint."
            ),
        )

    # Check image exists
    img_path = ds.get_image_path(body.sample_id)
    if img_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Image file for sample '{body.sample_id}' not found on disk.",
        )

    # Run inference
    try:
        result = infer.classify(
            model_id=body.model,
            model_info=m_info,
            sample_split=rec.split,
            sample_split_index=rec.split_index,
            sample_image_path=img_path,
            prefer_live=body.prefer_live,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {type(e).__name__}: {e}",
        )

    # Build top predictions list (only for DL models)
    top_predictions = [
        TopPrediction(
            rank=i + 1,
            class_id=nasa_id,
            class_name=cname,
            probability=round(prob, 6),
        )
        for i, (nasa_id, cname, prob) in enumerate(result.top_k)
    ]

    # Confidence display string
    if result.confidence is not None:
        conf_display = f"{result.confidence * 100:.2f}%"
    else:
        conf_display = "NOT AVAILABLE"

    # Inference time display string
    if result.inference_time_ms is not None:
        time_display = f"{result.inference_time_ms:.1f} ms"
    else:
        time_display = "N/A — STORED PREDICTION"

    # Correct / Incorrect
    is_correct = result.predicted_nasa_id == rec.ground_truth_id

    # Denormalized benchmark for the result panel
    bench = ms.get_model_performance_by_id(body.model)

    return ClassifyResponse(
        sample_id=body.sample_id,
        model=body.model,
        model_name=m_info["display_name"],
        predicted_class_id=result.predicted_nasa_id,
        predicted_class_name=result.predicted_class_name,
        ground_truth_id=rec.ground_truth_id,
        ground_truth_name=rec.ground_truth_name,
        correct=is_correct,
        confidence=result.confidence,
        confidence_display=conf_display,
        top_predictions=top_predictions,
        inference_source=result.inference_source,
        inference_time_ms=result.inference_time_ms,
        inference_time_display=time_display,
        status="success",
        model_test_metrics=bench,
    )
