"""Batch feature extraction pipeline extracting 8,186 handcrafted features per image."""

import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from preprocessing.image_processing import preprocess_image
from preprocessing.feature_extraction import (
    extract_features_from_preprocessed_image,
    get_feature_names,
    get_feature_group_breakdown
)


def extract_split_features(
    split_name: str,
    records: List[Tuple[str, int]],
    calibrated_dir: Path = config.CALIBRATED_IMG_DIR
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 8,186-dimensional feature matrix and labels for a dataset split."""
    feature_list = []
    labels = []
    total = len(records)

    print(f"Extracting features for {split_name} ({total} images)...")
    t0 = time.time()

    for idx, (rel_path, lbl) in enumerate(records):
        fname = Path(rel_path).name
        full_path = calibrated_dir / fname
        preprocessed = preprocess_image(str(full_path), target_size=(config.STANDARD_IMAGE_WIDTH, config.STANDARD_IMAGE_HEIGHT))
        vec = extract_features_from_preprocessed_image(preprocessed)
        feature_list.append(vec)
        labels.append(lbl)

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            print(f"  [{split_name}] Processed {idx + 1}/{total} ({elapsed:.1f}s)")

    X = np.vstack(feature_list).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


def run_feature_extraction_pipeline(output_dir: Path = config.FEATURES_DIR) -> None:
    """Executes feature extraction across all three NASA splits and caches artifacts."""
    from deep_learning.datasets import load_split_records
    train_records = load_split_records(config.TRAIN_LABELS_PATH)
    val_records = load_split_records(config.VAL_LABELS_PATH)
    test_records = load_split_records(config.TEST_LABELS_PATH)

    output_dir.mkdir(parents=True, exist_ok=True)

    for s_name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        X, y = extract_split_features(s_name, records)
        np.savez_compressed(output_dir / f"X_{s_name}.npz", X=X)
        np.save(output_dir / f"y_{s_name}.npy", y)
        print(f"  ✓ Saved {output_dir / f'X_{s_name}.npz'} and y_{s_name}.npy (shape: {X.shape})")

    names = get_feature_names()
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(names, f, indent=2)

    breakdown = get_feature_group_breakdown()
    with open(output_dir / "feature_metadata.json", "w") as f:
        json.dump({"total_features": len(names), "breakdown": breakdown}, f, indent=2)
    print("Feature extraction pipeline complete.")
