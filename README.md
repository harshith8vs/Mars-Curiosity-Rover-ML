# Automated Classification of Martian Surface Images Captured by NASA's Curiosity Rover Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()

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
- **Active Split Classes:** 24 active classes across splits.
- **Zero-Instance Class:** Class 22 (`sun`) has 0 instances in calibrated browse splits, representing an operational ground truth preserved without synthetic mutation.
- **Split Disjointness:** 100% strictly disjoint sets (0 overlapping images across train, val, and test; zero data leakage).

---

## 3. Preprocessing Pipeline
Two distinct, specialized preprocessing pipelines are provided:

### Classical Machine Learning (Handcrafted Features)
1. **Aspect-Ratio Preserving Letterbox:** Resizes raw browse images to standard 256×256 pixels with constant neutral black padding.
2. **Multi-Domain Feature Extraction (8,186 features):**
   - **Color Histograms:** RGB (3×32 bins) and HSV (32+16+16 bins).
   - **Texture Features:** Gray-Level Co-occurrence Matrix (GLCM) contrast, dissimilarity, homogeneity, energy, correlation, and ASM across 4 angles.
   - **Shape Features:** Log-transformed Hu Moments.
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

## 5. Official Model Benchmark Comparison

| Model | Accuracy | Macro-F1 | Evaluation Split | Model Category | Notes |
|:------|:--------:|:--------:|:----------------:|:---------------|:------|
| **KNN** | 50.85% | 0.6613 | Validation Split | Classical ML | Scaled + PCA-100, Distance Weighted |
| **Naive Bayes** | 12.07% | 0.2012 | Validation Split | Classical ML | Scaled, Uniform Class Priors |
| **Decision Tree** | 15.30% | 0.2690 | Validation Split | Classical ML | Unscaled, Balanced Class Weights |
| **Random Forest** | 66.04% | 0.5993 | Validation Split | Classical ML | Unscaled, 100 Trees, Balanced |
| **SVM** | 67.01% | 0.6393 | Validation Split | Classical ML | Scaled + PCA-100, RBF, Balanced |
| **MRSCAtt** | 64.29% | 0.5851 | Official Test Split | Deep Learning | Spatial & Channel Attention CNN |
| **ViT-B/16** | 73.26% | 0.6536 | Official Test Split | Deep Learning | 196 Patches of 16×16, 86M Params |
| **ResNet-50** | 77.78% | 0.6631 | Official Test Split | Deep Learning | Deep Learning Baseline (224×224) |
| **EfficientNet-B3** | **80.61%** | **0.7083** | Official Test Split | Deep Learning | **★ Best-Performing Model (Champion) ★** |

> [!IMPORTANT]
> **EfficientNet-B3 is currently the champion model**, achieving **80.61% Test Accuracy**, **0.7083 Test Macro-F1**, and **0.8288 Weighted-F1** across 1,305 test samples. The model checkpoint resides at `saved_models/deep_learning/efficientnet_b3/best_efficientnet_b3.pth` and raw predictions at `results/09_EfficientNet_B3/predictions/`.

### Methodological Notes on Evaluation Protocol

1. **Evaluation Split Asymmetry in Master Benchmark:**
   - KNN, Naive Bayes, Decision Tree, Random Forest, and SVM benchmark values in the master comparison are based on the **validation split (1,640 samples)**.
   - MRSCAtt, ViT-B/16, ResNet-50, and EfficientNet-B3 benchmark values are based on the **official test split (1,305 samples)**.
   - This asymmetry reflects the project's historical experimental workflow (classical models selected champion configurations based on validation F1 before deep learning baselines were trained).
   - The classical models also have test-set evaluation artifacts available separately in `results/01_KNN/` through `results/05_SVM/` (e.g., KNN test accuracy is 47.20% and SVM is 48.89%), but the canonical master benchmark comparison remains unchanged.

2. **Macro-F1 Calculation Across 24 Active Classes:**
   - The project evaluates Macro-F1 strictly over all 24 active classes.
   - The implementation explicitly specifies:
     ```python
     labels = list(range(24))
     macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
     ```
   - Two active classes have zero ground-truth samples in the official NASA test split:
     - NASA Class ID 5: `drill holes` (Active index 5)
     - NASA Class ID 23: `turret` (Active index 22)
   - Therefore, the Macro-F1 denominator remains 24 classes, and `zero_division=0` assigns F1 = 0.0 to zero-support classes. Averaging over all 24 classes yields ResNet-50 Macro-F1 = 0.6631 (whereas averaging only over non-zero support classes would yield 0.6919). This explicit protocol ensures 100% reproducibility.

---

## 6. Project Directory Structure

```
Mars ML models/
├── config.py                      # Centralized, dynamic project paths & configurations
├── main.py                        # Unified project CLI orchestrator
├── requirements.txt               # Documented project dependencies
├── README.md                      # Comprehensive academic project documentation
├── CLEANUP_REPORT.md              # Evidence-based codebase audit & refactoring report
├── .gitignore                     # Git exclusion rules
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
│   ├── training.py                # DeepLearningPipeline & fine-tuning engine
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
│   ├── feature_extraction.py      # Color, GLCM, Hu, HOG feature extractors (8,186 dims)
│   └── eda.py                     # Exploratory Data Analysis & audit helpers
│
├── features/                      # Extracted classical feature matrices
│   ├── X_train.npz, y_train.npy   # Training features (3,746 × 8,186)
│   ├── X_val.npz, y_val.npy       # Validation features (1,640 × 8,186)
│   ├── X_test.npz, y_test.npy     # Test features (1,305 × 8,186)
│   ├── feature_names.json         # Names of all 8,186 feature dimensions
│   └── extract.py                 # Batch feature extraction driver
│
├── saved_models/                  # Trained checkpoints (100% preserved)
│   ├── knn/                       # KNN-1 through KNN-4 .joblib models
│   ├── naive_bayes/               # NB-1 and NB-2 .joblib models
│   ├── decision_tree/             # DT-1 and DT-2 .joblib models
│   ├── random_forest/             # RF-1 and RF-2 .joblib models
│   ├── svm/                       # SVM-1 through SVM-4 .joblib models
│   └── deep_learning/             # PyTorch checkpoints (best & last)
│       ├── resnet50/              # best_resnet50.pth, last_resnet50.pth
│       ├── mrscatt/               # best_mrscatt.pth, last_mrscatt.pth
│       ├── vit/                   # best_vit.pth, last_vit.pth
│       └── efficientnet_b3/       # best_efficientnet_b3.pth, last_efficientnet_b3.pth
│
├── results/                       # Model-wise results hierarchy
│   ├── 01_KNN/                    # Reports, raw & normalized confusion matrices, predictions, metrics
│   ├── 02_Naive_Bayes/            # Reports, confusion matrices, predictions, metrics
│   ├── 03_Decision_Tree/          # Reports, confusion matrices, predictions, metrics
│   ├── 04_Random_Forest/          # Reports, confusion matrices, predictions, metrics
│   ├── 05_SVM/                    # Reports, confusion matrices, predictions, metrics
│   ├── 06_MRSCAtt/                # Reports, confusion matrices, curves, predictions, metrics
│   ├── 07_ViT_B16/                # Reports, confusion matrices, curves, predictions, metrics
│   ├── 08_ResNet50/               # Reports, confusion matrices, curves, predictions, metrics
│   ├── 09_EfficientNet_B3/        # Reports, confusion matrices, curves, predictions, metrics
│   ├── 10_Overall_Comparison/     # model_comparison.csv & model_comparison.png
│   └── 99_Unclassified/           # Dataset-wide EDA plots (6 files)
│
├── scripts/                       # Automation and batch execution utilities
│   └── run_all_models.py          # Batch re-execution and artifact regeneration
│
└── notebooks/                     # Interactive research documentation (10 curated notebooks)
    ├── 01_data_exploration.ipynb
    ├── 02_image_processing.ipynb
    ├── 03_feature_extraction.ipynb
    ├── 04a_imbalance_strategy.ipynb
    ├── 04b_five_ml_models.ipynb
    ├── 05_model_comparison.ipynb
    ├── 05a_resnet50.ipynb
    ├── 05b_mrscatt.ipynb
    ├── 05c_vit.ipynb
    └── 05d_efficientnet_b3.ipynb
```

---

## 7. How to Run the Project

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Display Academic Benchmark Table
```bash
python main.py --benchmark
```

### Verify NASA Dataset Integrity & Data Disjointness
```bash
python main.py --verify-dataset
```

### Run and Regenerate Results for All 9 Models
```bash
# Option A: Via Unified CLI
python main.py --regenerate-results

# Option B: Via Batch Script
python scripts/run_all_models.py
```

### Fast Evaluation Across All 9 Models
```bash
python main.py --evaluate-all
```

### Evaluate a Specific Model
```bash
# EfficientNet-B3 (Champion Model)
python main.py --model efficientnet_b3 --mode evaluate

# ResNet-50 Baseline
python main.py --model resnet50 --mode evaluate

# K-Nearest Neighbors (Classical Champion)
python main.py --model knn --mode evaluate
```

---

## 8. Where Artifacts are Stored
- **Results:** Organized numerically by model under `results/01_KNN` through `results/09_EfficientNet_B3`, multi-model comparisons in `results/10_Overall_Comparison`, and EDA plots in `results/99_Unclassified/eda_plots`.
- **Checkpoints:** Saved under `saved_models/` organized by model family.
- **Extracted Feature Matrices:** Stored in `features/` as compressed `.npz` and `.npy` arrays.
- **Audit & History:** Documented in `CLEANUP_REPORT.md`.
