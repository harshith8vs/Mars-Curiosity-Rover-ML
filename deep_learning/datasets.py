"""Dataset loading, split parsing, class mapping, and weight calculation for PyTorch."""

from collections import Counter
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import config


def load_synset_mapping(mapping_path: Path = config.CLASS_MAPPING_PATH) -> Dict[int, str]:
    """Load NASA synset word mapping returning dict of class ID to name."""
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Synset mapping file not found at: {mapping_path}")
        
    class_names: Dict[int, str] = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean:
                parts = clean.split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    class_names[int(parts[0])] = parts[1].strip()
                    
    assert len(class_names) == config.NUM_EXPECTED_CLASSES, (
        f"Expected {config.NUM_EXPECTED_CLASSES} classes in synset file, found {len(class_names)}"
    )
    return class_names


def load_split_records(split_path: Path) -> List[Tuple[str, int]]:
    """Parse NASA split file into list of (relative_path, label) tuples."""
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
        
    records: List[Tuple[str, int]] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            clean = line.strip()
            if not clean:
                continue
            parts = clean.split()
            if len(parts) != 2:
                raise ValueError(f"Malformed line {line_num} in {split_path}: '{line}'")
            img_rel = parts[0]
            label = int(parts[1])
            records.append((img_rel, label))
            
    return records


def build_active_class_mappings(
    train_records: List[Tuple[str, int]],
    class_names: Dict[int, str]
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str], List[int]]:
    """Build bidirectional mappings between NASA IDs and active indices (excluding class 22 'sun')."""
    train_labels = [lbl for _, lbl in train_records]
    active_classes = sorted(list(set(train_labels)))
    
    assert len(active_classes) == 24, (
        f"Expected exactly 24 active classes in training set, found {len(active_classes)}: {active_classes}"
    )
    assert config.ZERO_INSTANCE_CLASS_ID not in active_classes, (
        f"Class {config.ZERO_INSTANCE_CLASS_ID} ('sun') unexpectedly found in training split!"
    )
    
    original_to_active: Dict[int, int] = {orig: idx for idx, orig in enumerate(active_classes)}
    active_to_original: Dict[int, int] = {idx: orig for idx, orig in enumerate(active_classes)}
    active_to_name: Dict[int, str] = {idx: class_names[orig] for idx, orig in enumerate(active_classes)}
    
    return original_to_active, active_to_original, active_to_name, active_classes


def compute_training_class_weights(
    train_records: List[Tuple[str, int]],
    original_to_active: Dict[int, int]
) -> torch.Tensor:
    """Compute balanced class weights w_c = N / (K * N_c) directly from training labels."""
    active_labels = [original_to_active[lbl] for _, lbl in train_records]
    counts = Counter(active_labels)
    
    num_samples = len(train_records)
    num_classes = len(original_to_active)
    
    weights = np.zeros(num_classes, dtype=np.float32)
    for idx in range(num_classes):
        c_count = counts.get(idx, 0)
        if c_count == 0:
            raise ValueError(f"Active class index {idx} has 0 training samples!")
        weights[idx] = num_samples / (num_classes * c_count)
        
    return torch.tensor(weights, dtype=torch.float32)


class MarsRoverDataset(Dataset):
    """PyTorch Dataset for loading calibrated Mars rover images with active labels."""
    
    def __init__(
        self,
        records: List[Tuple[str, int]],
        calibrated_dir: Path,
        original_to_active: Dict[int, int],
        transform: Optional[Callable] = None,
        verify_files: bool = False
    ):
        self.records = records
        self.calibrated_dir = Path(calibrated_dir)
        self.original_to_active = original_to_active
        self.transform = transform
        
        if verify_files:
            for img_rel, _ in self.records:
                fname = Path(img_rel).name
                full_path = self.calibrated_dir / fname
                if not full_path.is_file():
                    raise FileNotFoundError(f"Image not found: {full_path}")
                    
    def __len__(self) -> int:
        return len(self.records)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_rel, original_label = self.records[idx]
        fname = Path(img_rel).name
        img_path = self.calibrated_dir / fname
        
        # Load image via PIL and ensure 3-channel RGB
        try:
            with Image.open(img_path) as pil_img:
                image = pil_img.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image {img_path}: {e}") from e
            
        if self.transform is not None:
            image = self.transform(image)
            
        target = self.original_to_active[original_label]
        return image, target
        
    def get_metadata(self, idx: int) -> Dict[str, object]:
        """Return metadata for a given sample index."""
        img_rel, original_label = self.records[idx]
        return {
            "index": idx,
            "filename": Path(img_rel).name,
            "relative_path": img_rel,
            "original_label": original_label,
            "active_label": self.original_to_active[original_label]
        }
