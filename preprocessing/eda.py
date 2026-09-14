"""Exploratory Data Analysis (EDA) utilities for dataset distribution and image auditing."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import config


def load_all_split_data() -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]], Dict[int, str]]:
    """Loads all split records and class mapping from config paths."""
    from deep_learning.datasets import load_split_records, load_synset_mapping
    train = load_split_records(config.TRAIN_LABELS_PATH)
    val = load_split_records(config.VAL_LABELS_PATH)
    test = load_split_records(config.TEST_LABELS_PATH)
    mapping = load_synset_mapping(config.CLASS_MAPPING_PATH)
    return train, val, test, mapping


def audit_image_dimensions(sample_size: Optional[int] = 100) -> Dict[str, Any]:
    """Audits image dimensions, formats, and channel modes across dataset splits."""
    train, val, test, _ = load_all_split_data()
    all_records = train + val + test
    if sample_size and sample_size < len(all_records):
        import random
        random.seed(42)
        records = random.sample(all_records, sample_size)
    else:
        records = all_records

    dimensions = Counter()
    modes = Counter()
    formats = Counter()

    for rel_path, _ in records:
        fname = Path(rel_path).name
        fpath = config.CALIBRATED_IMG_DIR / fname
        if not fpath.is_file():
            continue
        with Image.open(fpath) as img:
            dimensions[img.size] += 1
            modes[img.mode] += 1
            formats[img.format] += 1

    return {
        "sampled_images": len(records),
        "dimensions": dict(dimensions),
        "color_modes": dict(modes),
        "formats": dict(formats)
    }


def audit_class_distribution() -> Dict[str, Any]:
    """Audits class counts and imbalance ratio across train, val, test splits."""
    train, val, test, mapping = load_all_split_data()
    train_c = Counter(lbl for _, lbl in train)
    val_c = Counter(lbl for _, lbl in val)
    test_c = Counter(lbl for _, lbl in test)
    total_c = train_c + val_c + test_c

    active_counts = [cnt for cid, cnt in train_c.items() if cnt > 0]
    imbalance_ratio = max(active_counts) / min(active_counts) if active_counts else 0.0

    return {
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "total_samples": len(train) + len(val) + len(test),
        "active_classes_count": len(active_counts),
        "zero_instance_class": config.ZERO_INSTANCE_CLASS_ID,
        "imbalance_ratio": float(imbalance_ratio),
        "train_counts": dict(train_c),
        "val_counts": dict(val_c),
        "test_counts": dict(test_c)
    }
