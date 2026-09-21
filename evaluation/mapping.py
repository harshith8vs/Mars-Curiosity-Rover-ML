"""
Canonical class mapping module for the NASA Mars Curiosity Rover dataset.

This module is the single source of truth for:
  - Active NASA class IDs (24 classes, excluding class 22 "sun")
  - Bidirectional mappings between NASA IDs and active DL indices (0–23)
  - Human-readable class names

Key facts:
  - The NASA dataset defines 25 classes (IDs 0–24).
  - Class 22 ("sun") has ZERO samples in ALL official splits (train, val, test).
    It is therefore GLOBALLY INACTIVE and excluded from the 24 active classes.
  - All other 24 classes (IDs 0–21, 23, 24) are ACTIVE.
  - NASA IDs 5 ("drill holes") and 23 ("turret") are active classes but happen to
    have ZERO ground-truth samples specifically in the official TEST split (1,305 images).
    They must NOT be excluded from the Macro-F1 denominator; zero_division=0 assigns
    F1 = 0.0 to zero-support classes, and they still count in the average.
  - The deep-learning models remap NASA IDs to contiguous indices 0–23
    by skipping the absent class 22. The classical ML models use raw NASA IDs.
"""

from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# All 25 NASA class IDs and their names
# ──────────────────────────────────────────────────────────────────────────────
NASA_CLASS_NAMES: Dict[int, str] = {
    0:  "apxs",
    1:  "apxs cal target",
    2:  "chemcam cal target",
    3:  "chemin inlet open",
    4:  "drill",
    5:  "drill holes",
    6:  "drt front",
    7:  "drt side",
    8:  "ground",
    9:  "horizon",
    10: "inlet",
    11: "mahli",
    12: "mahli cal target",
    13: "mastcam",
    14: "mastcam cal target",
    15: "observation tray",
    16: "portion box",
    17: "portion tube",
    18: "portion tube opening",
    19: "rems uv sensor",
    20: "rover rear deck",
    21: "scoop",
    22: "sun",        # INACTIVE — zero samples in all official splits
    23: "turret",
    24: "wheel",
}

# ──────────────────────────────────────────────────────────────────────────────
# The one globally inactive class
# ──────────────────────────────────────────────────────────────────────────────
INACTIVE_NASA_CLASS_ID: int = 22            # "sun" — absent from all splits
INACTIVE_NASA_CLASS_NAME: str = "sun"

# ──────────────────────────────────────────────────────────────────────────────
# 24 ACTIVE NASA class IDs (ordered, class 22 excluded)
# Used by classical ML models (which keep raw NASA IDs as labels)
# ──────────────────────────────────────────────────────────────────────────────
ACTIVE_NASA_CLASS_IDS: List[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 23, 24
]
NUM_ACTIVE_CLASSES: int = 24
assert len(ACTIVE_NASA_CLASS_IDS) == NUM_ACTIVE_CLASSES, "Sanity: 24 active classes"
assert INACTIVE_NASA_CLASS_ID not in ACTIVE_NASA_CLASS_IDS, "Sun must be excluded"

# ──────────────────────────────────────────────────────────────────────────────
# 24 contiguous active indices 0–23
# Used by deep-learning models (which remap class IDs to 0-based contiguous range)
# ──────────────────────────────────────────────────────────────────────────────
ACTIVE_INDICES: List[int] = list(range(NUM_ACTIVE_CLASSES))   # [0, 1, …, 23]

# ──────────────────────────────────────────────────────────────────────────────
# Classes that have ZERO samples in the official TEST split (1,305 images)
# These are STILL ACTIVE classes; they must remain in the Macro-F1 denominator.
# zero_division=0 assigns F1 = 0.0 for them, and the average is taken over all 24.
# ──────────────────────────────────────────────────────────────────────────────
ZERO_SUPPORT_IN_TEST: List[int] = [5, 23]   # NASA IDs (drill holes, turret)
# Corresponding DL active indices (NASA 5 → index 5, NASA 23 → index 22)
ZERO_SUPPORT_IN_TEST_DL_INDICES: List[int] = [5, 22]

# ──────────────────────────────────────────────────────────────────────────────
# Bidirectional mappings: NASA ID ↔ DL active index
# ──────────────────────────────────────────────────────────────────────────────
NASA_TO_ACTIVE_INDEX: Dict[int, int] = {
    nasa_id: idx for idx, nasa_id in enumerate(ACTIVE_NASA_CLASS_IDS)
}
ACTIVE_INDEX_TO_NASA: Dict[int, int] = {
    idx: nasa_id for idx, nasa_id in enumerate(ACTIVE_NASA_CLASS_IDS)
}

# Human-readable names keyed by DL active index
ACTIVE_INDEX_TO_CLASS_NAME: Dict[int, str] = {
    idx: NASA_CLASS_NAMES[nasa_id]
    for idx, nasa_id in enumerate(ACTIVE_NASA_CLASS_IDS)
}

# Human-readable names keyed by NASA ID (active only)
NASA_ID_TO_CLASS_NAME: Dict[int, str] = {
    nasa_id: NASA_CLASS_NAMES[nasa_id]
    for nasa_id in ACTIVE_NASA_CLASS_IDS
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def nasa_id_to_active_index(nasa_id: int) -> int:
    """Convert a raw NASA class ID to the DL active index (0–23)."""
    if nasa_id not in NASA_TO_ACTIVE_INDEX:
        raise ValueError(
            f"NASA ID {nasa_id} is not an active class. "
            f"Active IDs: {ACTIVE_NASA_CLASS_IDS}"
        )
    return NASA_TO_ACTIVE_INDEX[nasa_id]


def active_index_to_nasa_id(active_idx: int) -> int:
    """Convert a DL active index (0–23) to the raw NASA class ID."""
    if active_idx not in ACTIVE_INDEX_TO_NASA:
        raise ValueError(
            f"Active index {active_idx} out of range [0, {NUM_ACTIVE_CLASSES-1}]."
        )
    return ACTIVE_INDEX_TO_NASA[active_idx]


def get_class_name_by_nasa_id(nasa_id: int) -> str:
    """Return human-readable class name for a NASA class ID."""
    return NASA_CLASS_NAMES.get(nasa_id, f"unknown_class_{nasa_id}")


def get_class_name_by_active_index(active_idx: int) -> str:
    """Return human-readable class name for a DL active index."""
    return ACTIVE_INDEX_TO_CLASS_NAME.get(active_idx, f"unknown_idx_{active_idx}")


def to_active_index_array(nasa_labels):
    """Convert an array of NASA IDs to an array of DL active indices."""
    import numpy as np
    return np.array([NASA_TO_ACTIVE_INDEX[int(x)] for x in nasa_labels], dtype=np.int64)


def to_nasa_id_array(active_labels):
    """Convert an array of DL active indices to an array of NASA IDs."""
    import numpy as np
    return np.array([ACTIVE_INDEX_TO_NASA[int(x)] for x in active_labels], dtype=np.int64)


def validate_active_class_ids():
    """Assert internal consistency of the mapping module."""
    assert len(ACTIVE_NASA_CLASS_IDS) == NUM_ACTIVE_CLASSES
    assert INACTIVE_NASA_CLASS_ID not in ACTIVE_NASA_CLASS_IDS
    assert len(NASA_TO_ACTIVE_INDEX) == NUM_ACTIVE_CLASSES
    assert len(ACTIVE_INDEX_TO_NASA) == NUM_ACTIVE_CLASSES
    assert len(ACTIVE_INDEX_TO_CLASS_NAME) == NUM_ACTIVE_CLASSES
    assert len(ACTIVE_INDICES) == NUM_ACTIVE_CLASSES
    # Round-trip check
    for idx, nasa_id in enumerate(ACTIVE_NASA_CLASS_IDS):
        assert NASA_TO_ACTIVE_INDEX[nasa_id] == idx
        assert ACTIVE_INDEX_TO_NASA[idx] == nasa_id
    return True


# Run sanity check on import
validate_active_class_ids()
