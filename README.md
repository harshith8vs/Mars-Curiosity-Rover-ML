# Automated Classification of Martian Surface Images Captured by NASA's Curiosity Rover Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()
[![Tests: 52 passed](https://img.shields.io/badge/tests-52%20passed-brightgreen.svg)]()

A modular, reproducible machine learning and deep learning research benchmark for the automated geological and engineering classification of Martian surface terrain images captured by NASA's Mars Science Laboratory (MSL) Curiosity rover.

---

## 1. Problem Statement
The NASA Curiosity rover transmits thousands of multi-spectral and monochromatic surface images to Earth via Deep Space Network relays with constrained orbital communication windows. Autonomous and ground-based computer vision systems must reliably classify Martian surface features (e.g., bedrock, drill holes, wheel tracks, sand dunes, arm joints, horizons) under extreme real-world challenges:
- **Severe Class Imbalance:** The natural operational class distribution spans ~122× imbalance across active terrain and rover classes.
- **Extreme Domain Variation:** Varying solar angles, atmospheric dust opacity, Mastcam focal lengths, and camera sensor artifacts.
- **Zero Data Leakage:** Strict operational requirement that normalization, feature scaling, and dimensionality reduction must never leak validation or test statistics into training.

---

## 2. Dataset
The benchmark uses NASA's official dataset of 6,691 labeled browse images, partitioned into 3,746 training, 1,640 validation, and 1,305 test samples across 24 active classes.

> [!NOTE]
> The calibrated archive directory physically contains 6,737 JPG files. The additional 46 files are unindexed browse images present in the archive but excluded from NASA's published partition files. They are not used for training, validation, testing, or feature extraction.

- **Total Partitioned Images:** 6,691 verified calibrated browse images.
  - **Training Split:** 3,746 images (`train-calibrated-shuffled.txt`)
  - **Validation Split:** 1,640 images (`val-calibrated-shuffled.txt`)
  - **Testing Split:** 1,305 images (`test-calibrated-shuffled.txt`)
- **Defined Classes:** 25 synset classes (IDs 0–24) documented in `msl_synset_words-indexed.txt`.
- **Active Classes:** 24 active classes. Class 22 (`sun`) is **globally inactive** (0 instances in all splits).
- **Zero-Support Classes in TEST:** NASA IDs 5 (`drill holes`) and 23 (`turret`) are active classes but have **0 ground-truth samples specifically in the official TEST split**. They remain in the 24-class Macro-F1 denominator; `zero_division=0` assigns F1=0.0 to them.
- **Split Disjointness:** 100% strictly disjoint sets (0 overlapping images across train, val, and test; zero data leakage).

---

## 3. Preprocessing Pipeline
Two distinct, specialized preprocessing pipelines are provided:

### Classical Machine Learning (Handcrafted Features)
1. **Aspect-Ratio Preserving Letterbox:** Resizes raw browse images to standard 256×256 pixels with constant neutral black padding.
2. **Multi-Domain Feature Extraction (8,186 features):**
   - **Color Histograms:** RGB (3×32 bins).
   - **Texture Features:** Gray-Level Co-occurrence Matrix (GLCM) contrast, dissimilarity, homogeneity, energy, correlation, and ASM across 4 angles.
   - **Structural / Edge Features:** Histogram of Oriented Gradients (HOG, 8,100 features) + Sobel spatial edge statistics.
3. **Leakage-Safe Scaling:** `TrainingScaler` fits `StandardScaler` exclusively on training feature vectors; transforms val and test sets without statistical drift.
4. **PCA Dimensionality Reduction:** `PCA(n_components=100)` fitted strictly on `X_train_scaled`.

### Deep Learning (End-to-End Raw Pixels)
1. **LetterboxResize:** 224×224 (ResNet-50, MRSCAtt, ViT) or 300×300 (EfficientNet-B3) with neutral padding.
2. **Conservative Data Augmentation:** Moderate random crops, subtle rotation (±10°), mild color jitter, and horizontal flips.
3. **ImageNet Normalization:** Channel-wise mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

---

## 4. Models
The repository evaluates 9 distinct models across classical and deep learning paradigms:

1. **K-Nearest Neighbors (KNN):** Euclidean distance, distance-weighted nearest neighbors on PCA-100 features.
2. **Gaussian Naive Bayes:** Scaled feature vectors with uniform class priors.
3. **Decision Tree:** Unscaled raw features with balanced class weights.
4. **Random Forest:** Ensemble of 100 decision trees with balanced class weighting.
5. **Support Vector Machine (SVM):** RBF kernel on PCA-100 features with balanced class penalties.
6. **MRSCAtt:** Pretrained ResNet-50 backbone augmented with a custom `ChannelSpatialBlock` (dual-branch CBAM channel attention + 7×7 spatial attention) placed after `layer4`.
7. **ViT-B/16:** Vision Transformer operating on 196 non-overlapping 16×16 patches + class token.
8. **ResNet-50:** Two-stage fine-tuned 50-layer deep residual network baseline.
9. **EfficientNet-B3:** Compound scaled convolutional architecture with depth, width, and resolution scaling (300×300 input).

---

## 5. Evaluation Protocol

### Two-Phase Design
The project follows a strict two-phase evaluation protocol to prevent test-set contamination during model selection.

**Phase A — Model Selection (Validation Split):**
Each classical ML family is evaluated over multiple hyperparameter configurations. The champion configuration per family is selected using **Validation Macro-F1** as the primary metric. The test set is never consulted during this phase.

**Phase B — Final Cross-Model Comparison (TEST Split):**
All 9 final champion models are evaluated **once** on the official NASA TEST split (1,305 images). Metrics are computed using the canonical 24-class protocol. This is the definitive benchmark for all reported results.

### Canonical 24-Class Macro-F1 Protocol

All reported metrics use explicit active-class label lists:

```python
# Classical ML (raw NASA IDs):
from evaluation.mapping import ACTIVE_NASA_CLASS_IDS
macro_f1 = f1_score(y_true, y_pred, labels=ACTIVE_NASA_CLASS_IDS, average="macro", zero_division=0)

# Deep Learning (remapped active indices 0–23):
from evaluation.mapping import ACTIVE_INDICES
macro_f1 = f1_score(y_true, y_pred, labels=ACTIVE_INDICES, average="macro", zero_division=0)
```

- Class 22 (`sun`) is **excluded** from all label lists (globally inactive).
- NASA IDs 5 and 23 are included in the label list (active) but have 0 test samples → F1=0.0 counted toward the 24-class average.

---

## 6. Official Final TEST Set Benchmark (Phase B)

> [!IMPORTANT]
> All metrics are computed on the **official NASA TEST split (1,305 samples)** using the canonical 24-class protocol.  
> **No checkpoint files are required** to reproduce these results — use the stored prediction arrays in `results/`.

| # | Model | Category | Accuracy (%) | Macro-F1 | Macro-P | Macro-R | Weighted-F1 |
|---|-------|----------|:------------:|:--------:|:-------:|:-------:|:-----------:|
| 1 | KNN (Scaled+PCA) | Classical | 47.20 | 0.5182 | 0.6807 | 0.4852 | 0.4386 |
| 2 | Naive Bayes (Scaled) | Classical | 18.01 | 0.2093 | 0.3280 | 0.2249 | 0.1848 |
| 3 | Decision Tree (Balanced) | Classical | 25.44 | 0.2551 | 0.2957 | 0.2997 | 0.2507 |
| 4 | Random Forest (Balanced) | Classical | 50.42 | 0.4305 | 0.5527 | 0.4685 | 0.4809 |
| 5 | SVM (Scaled+PCA, Balanced) | Classical | 48.89 | 0.4920 | 0.6140 | 0.4842 | 0.4530 |
| 6 | MRSCAtt (Attention) | Deep Learning | 64.29 | 0.5851 | 0.6461 | 0.6553 | 0.6515 |
| 7 | ViT-B/16 (Transformer) | Deep Learning | 73.26 | 0.6536 | 0.7182 | 0.6988 | 0.7505 |
| 8 | ResNet-50 (Baseline) | Deep Learning | 77.78 | 0.6631 | 0.6886 | 0.7273 | 0.7937 |
| 9 | **EfficientNet-B3 (Champion)** | **Deep Learning** | **80.61** | **0.7083** | **0.7307** | **0.7487** | **0.8288** |

> EfficientNet-B3 achieves the best accuracy (80.61%) and Macro-F1 (0.7083) across all 9 models.

---

## 7. Repository Policy: No Checkpoint Files

> [!IMPORTANT]
> **Trained model checkpoints (`.pth`, `.joblib`) are intentionally excluded from this repository** due to their large size (multi-GB total). The repository is designed to be lightweight.
>
> All benchmark results can be **fully reproduced without checkpoint files** using stored prediction arrays in `results/`. Use:
> ```bash
> python main.py --verify-results   # Recompute all canonical metrics from stored predictions
> python main.py --compare          # Display the full Phase B comparison table
> ```
>
> If you train models locally, checkpoints will be saved to `saved_models/` (gitignored).

---

## 8. Project Directory Structure

```
Mars ML models/
├── config.py                      # Centralized, dynamic project paths & configurations
├── main.py                        # Unified project CLI orchestrator
├── requirements.txt               # Documented project dependencies
├── README.md                      # Comprehensive academic project documentation
├── CLEANUP_REPORT.md              # Evidence-based codebase audit & refactoring report
├── .gitignore                     # Git exclusion rules (includes saved_models/, *.pth, *.joblib)
│
├── evaluation/                    # Canonical metric engine
│   ├── mapping.py                 # SINGLE SOURCE OF TRUTH: active class IDs, NASA↔DL bidirectional maps
│   └── metrics.py                 # compute_canonical_metrics() with explicit 24-class label lists
│
├── tests/                         # Automated validation tests (12 invariants, 52 test cases)
│   └── test_dataset_and_metrics.py
│
├── Mars rover data/               # NASA Curiosity official splits and calibrated images
│   ├── calibrated/                # 6,737 calibrated Martian browse images (.JPG)
│   ├── train-calibrated-shuffled.txt # Official training split records (3,746 samples)
│   ├── val-calibrated-shuffled.txt   # Official validation split records (1,640 samples)
│   ├── test-calibrated-shuffled.txt  # Official test split records (1,305 samples)
│   └── msl_synset_words-indexed.txt  # Official 25-class synset mapping
│
├── models/                        # Classical machine learning modules
│   ├── classical_pipeline.py      # Unified classical training, inference & evaluation engine
│   ├── knn.py                     # K-Nearest Neighbors classifier
│   ├── naive_bayes.py             # Gaussian Naive Bayes classifier
│   ├── decision_tree.py           # Decision Tree classifier
│   ├── random_forest.py           # Random Forest classifier
│   ├── svm.py                     # Support Vector Machine (SVC)
│   ├── imbalance.py               # Imbalance audit & class weight utilities
│   └── scaling.py                 # TrainingScaler & PCAStrategy
│
├── deep_learning/                 # Deep learning infrastructure
│   ├── datasets.py                # PyTorch Dataset & NASA label parser
│   ├── transforms.py              # LetterboxResize & normalization transforms
│   ├── training.py                # DeepLearningPipeline & architecture-aware fine-tuning engine
│   ├── evaluation.py              # Multiclass evaluation metrics & confusion matrix logic
│   ├── utils.py                   # Plotting, seeding, and export helpers
│   └── models/                    # Neural network architectures
│       ├── resnet50.py            # Pretrained ResNet-50
│       ├── mrscatt.py             # MRSCAtt with ChannelSpatialBlock
│       ├── vit.py                 # Vision Transformer (ViT-B/16)
│       └── efficientnet_b3.py     # EfficientNet-B3 compound scaled CNN
│
├── preprocessing/                 # Image processing & feature extraction
│   ├── image_processing.py        # Letterboxing, channel resolution, OpenCV utilities
│   ├── feature_extraction.py      # Color, GLCM, HOG feature extractors (8,186 dims)
│   └── eda.py                     # Exploratory Data Analysis & audit helpers
│
├── features/                      # Extracted classical feature matrices
│   ├── X_train.npz, y_train.npy   # Training features (3,746 × 8,186)
│   ├── X_val.npz, y_val.npy       # Validation features (1,640 × 8,186)
│   ├── X_test.npz, y_test.npy     # Test features (1,305 × 8,186)
│   ├── feature_names.json         # Names of all 8,186 feature dimensions
│   └── extract.py                 # Batch feature extraction driver
│
├── saved_models/                  # Trained checkpoints (LOCAL ONLY — gitignored)
│   ├── knn/                       # KNN-1 through KNN-4 .joblib models
│   ├── naive_bayes/               # NB-1 and NB-2 .joblib models
│   ├── decision_tree/             # DT-1 and DT-2 .joblib models
│   ├── random_forest/             # RF-1 and RF-2 .joblib models
│   ├── svm/                       # SVM-1 through SVM-4 .joblib models
│   └── deep_learning/             # PyTorch checkpoints (best & last) — NOT committed
│       ├── resnet50/              # best_resnet50.pth, last_resnet50.pth (local only)
│       ├── mrscatt/               # best_mrscatt.pth, last_mrscatt.pth (local only)
│       ├── vit/                   # best_vit.pth, last_vit.pth (local only)
│       └── efficientnet_b3/       # best_efficientnet_b3.pth (local only)
│
├── results/                       # Model-wise results hierarchy (committed, no checkpoints)
│   ├── 01_KNN/ … 05_SVM/          # Reports, confusion matrices, predictions, metrics
│   ├── 06_MRSCAtt/ … 09_EfficientNet_B3/  # Reports, confusion matrices, predictions
│   ├── 10_Overall_Comparison/     # final_test_comparison.{csv,json,md,png,heatmap}
│   └── 99_Unclassified/           # Dataset-wide EDA plots
│
└── scripts/                       # Automation and batch execution utilities
    ├── compare_all_models.py      # Phase B comparison: all 9 models, canonical metrics
    └── run_all_models.py          # Batch re-execution and artifact regeneration
```

---

## 9. How to Run the Project

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify Canonical Test Metrics (Checkpoint-Free — Fastest)
```bash
python main.py --verify-results
```

### Display Full Phase B Comparison Table
```bash
python main.py --compare
```

### Display Academic Benchmark Summary
```bash
python main.py --benchmark
```

### Verify NASA Dataset Integrity & Data Disjointness
```bash
python main.py --verify-dataset
```

### Run Automated Test Suite (12 Invariants)
```bash
python -m pytest tests/test_dataset_and_metrics.py -v
```

### Evaluate a Specific Model (Checkpoint-Free)
```bash
python main.py --model efficientnet_b3   # Champion model
python main.py --model resnet50          # ResNet-50 baseline
python main.py --model knn              # Classical KNN champion
```

### Regenerate All Artifacts (Requires Local Checkpoints)
```bash
# Option A: Via Unified CLI
python main.py --regenerate-results

# Option B: Via Batch Script
python scripts/run_all_models.py

# Option C: Phase B comparison only (checkpoint-free)
python scripts/compare_all_models.py
```

---

## 10. Where Artifacts are Stored
- **Results:** Organized numerically by model under `results/01_KNN` through `results/09_EfficientNet_B3`. Phase B comparison tables in `results/10_Overall_Comparison/`.
- **Checkpoints:** Saved locally under `saved_models/` (excluded from git). Not required for result verification.
- **Extracted Feature Matrices:** Stored in `features/` as compressed `.npz` and `.npy` arrays.
- **Audit & History:** Documented in `CLEANUP_REPORT.md`.
- **Canonical Class Mapping:** `evaluation/mapping.py` — single source of truth for all class ID operations.
