"""
Dataset service for the Mars Rover ML API.

Indexes all three official dataset splits (train, val, test) and provides:
  - Fast lookup by sample_id.
  - Keyword search, split filtering, class filtering, and pagination.
  - Safe absolute image-path resolution within 'Mars rover data/calibrated/'.
  - No dataset images are loaded eagerly; paths are resolved on demand.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evaluation.mapping import NASA_CLASS_NAMES, ACTIVE_NASA_CLASS_IDS


# ─────────────────────────────────────────────────────────────────────────────
# Internal sample record
# ─────────────────────────────────────────────────────────────────────────────

class _SampleRecord:
    __slots__ = (
        "sample_id", "filename", "split", "split_index",
        "ground_truth_id", "ground_truth_name", "abs_image_path",
    )

    def __init__(
        self,
        sample_id: str,
        filename: str,
        split: str,
        split_index: int,
        ground_truth_id: int,
        ground_truth_name: str,
        abs_image_path: str,
    ):
        self.sample_id = sample_id
        self.filename = filename
        self.split = split
        self.split_index = split_index
        self.ground_truth_id = ground_truth_id
        self.ground_truth_name = ground_truth_name
        self.abs_image_path = abs_image_path


# ─────────────────────────────────────────────────────────────────────────────
# DatasetService
# ─────────────────────────────────────────────────────────────────────────────

class DatasetService:
    """Thread-safe, read-only index of all dataset samples."""

    # Split names → (split_label, split_file_path, id_prefix)
    _SPLIT_DEFS = [
        ("test",  "test-calibrated-shuffled.txt",  "TEST"),
        ("val",   "val-calibrated-shuffled.txt",   "VAL"),
        ("train", "train-calibrated-shuffled.txt", "TRAIN"),
    ]

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._data_dir = project_root / "Mars rover data"
        self._calibrated_dir = self._data_dir / "calibrated"

        # Primary index: sample_id → _SampleRecord
        self._by_id: Dict[str, _SampleRecord] = {}

        # Secondary index: split → list of sample_ids (ordered)
        self._by_split: Dict[str, List[str]] = {"test": [], "val": [], "train": []}

        # Index of test split_index → sample_id (for stored prediction alignment)
        self._test_index_to_id: Dict[int, str] = {}

        self._load_splits()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_splits(self) -> None:
        """Parse all three split text files and populate internal indices."""
        for split_label, filename, id_prefix in self._SPLIT_DEFS:
            split_path = self._data_dir / filename
            if not split_path.exists():
                continue

            with open(split_path, "r", encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]

            # The header line (if any) starts without 'calibrated/'; skip it
            for split_index, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 2:
                    continue

                rel_path, nasa_id_str = parts[0], parts[1]
                try:
                    nasa_id = int(nasa_id_str)
                except ValueError:
                    continue

                # Filename is the basename of the relative path
                filename_bare = Path(rel_path).name
                abs_path = str((self._data_dir / rel_path).resolve())

                # Safety check: resolved path must stay within calibrated dir
                calibrated_abs = str(self._calibrated_dir.resolve())
                if not abs_path.startswith(calibrated_abs):
                    continue  # Skip any path traversal attempt

                sample_id = f"{id_prefix}-{split_index:04d}"
                class_name = NASA_CLASS_NAMES.get(nasa_id, f"class_{nasa_id}")

                record = _SampleRecord(
                    sample_id=sample_id,
                    filename=filename_bare,
                    split=split_label,
                    split_index=split_index,
                    ground_truth_id=nasa_id,
                    ground_truth_name=class_name,
                    abs_image_path=abs_path,
                )

                self._by_id[sample_id] = record
                self._by_split[split_label].append(sample_id)

                if split_label == "test":
                    self._test_index_to_id[split_index] = sample_id

    # ─────────────────────────────────────────────────────────────────────────
    # Public query API
    # ─────────────────────────────────────────────────────────────────────────

    def get_sample(self, sample_id: str) -> Optional[_SampleRecord]:
        return self._by_id.get(sample_id)

    def get_image_path(self, sample_id: str) -> Optional[str]:
        """Return the absolute filesystem path to the sample image, or None."""
        record = self._by_id.get(sample_id)
        if record is None:
            return None
        if not os.path.isfile(record.abs_image_path):
            return None
        return record.abs_image_path

    def get_test_sample_by_index(self, index: int) -> Optional[_SampleRecord]:
        """Retrieve test sample by its 0-based position in the test split file."""
        sid = self._test_index_to_id.get(index)
        return self._by_id.get(sid) if sid else None

    def counts(self) -> Dict[str, int]:
        return {
            "test": len(self._by_split["test"]),
            "val": len(self._by_split["val"]),
            "train": len(self._by_split["train"]),
            "total": len(self._by_id),
        }

    def get_stats(self) -> Dict:
        from evaluation.mapping import ACTIVE_NASA_CLASS_IDS, NASA_CLASS_NAMES

        class_counts = {}
        for cid in ACTIVE_NASA_CLASS_IDS:
            class_counts[cid] = {
                "class_id": str(cid),
                "class_name": NASA_CLASS_NAMES.get(cid, f"class_{cid}"),
                "train_count": 0,
                "val_count": 0,
                "test_count": 0,
                "total": 0,
            }

        for rec in self._by_id.values():
            cid = rec.ground_truth_id
            if cid in class_counts:
                class_counts[cid][f"{rec.split}_count"] += 1
                class_counts[cid]["total"] += 1

        sorted_classes = sorted(class_counts.values(), key=lambda x: int(x["class_id"]))
        return {
            "total_calibrated_images": 6737,
            "total_labeled_images": len(self._by_id),
            "active_classes": 24,
            "format": "JPEG",
            "preprocessed_size": "256 × 256",
            "color_modes": {"rgb": 6519, "grayscale": 172},
            "classical_feature_vector_dim": 8186,
            "feature_breakdown": {
                "color_statistics": 19,
                "rgb_histograms": 48,
                "glcm_texture": 12,
                "hog": 8100,
                "edge_descriptors": 2,
                "gradient_statistics": 5,
            },
            "splits": self.counts(),
            "classes": sorted_classes,
            "total_images": len(self._by_id),
        }

    def list_samples(
        self,
        *,
        split: Optional[str] = None,
        class_id: Optional[int] = None,
        query: Optional[str] = None,
        page: int = 1,
        page_size: int = 24,
    ) -> Tuple[List[_SampleRecord], int]:
        """
        Filter, search, and paginate samples.

        Returns (page_records, total_matching_count).
        page is 1-indexed.
        """
        # Determine which split IDs to search
        if split and split in self._by_split:
            candidate_ids = self._by_split[split]
        else:
            # Combine in canonical display order: test, val, train
            candidate_ids = (
                self._by_split["test"]
                + self._by_split["val"]
                + self._by_split["train"]
            )

        # Apply filters
        results: List[_SampleRecord] = []
        query_lower = query.lower() if query else None

        for sid in candidate_ids:
            rec = self._by_id[sid]

            # Class filter
            if class_id is not None and rec.ground_truth_id != class_id:
                continue

            # Keyword search on filename and class name
            if query_lower:
                haystack = (rec.filename + " " + rec.ground_truth_name + " " + sid).lower()
                if query_lower not in haystack:
                    continue

            results.append(rec)

        total = len(results)
        page_size = max(1, min(page_size, 100))
        page = max(1, page)
        start = (page - 1) * page_size
        end = start + page_size

        return results[start:end], total


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — initialized lazily in server.py
# ─────────────────────────────────────────────────────────────────────────────

_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """Return the module-level DatasetService singleton."""
    global _dataset_service
    if _dataset_service is None:
        project_root = Path(__file__).resolve().parent.parent
        _dataset_service = DatasetService(project_root)
    return _dataset_service
