"""Central configuration module defining dynamic project paths and dataset constants."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset paths
DATA_DIR = PROJECT_ROOT / "Mars rover data"
CALIBRATED_IMG_DIR = DATA_DIR / "calibrated"

# NASA official split files and documentation
TRAIN_LABELS_PATH = DATA_DIR / "train-calibrated-shuffled.txt"
VAL_LABELS_PATH = DATA_DIR / "val-calibrated-shuffled.txt"
TEST_LABELS_PATH = DATA_DIR / "test-calibrated-shuffled.txt"
CLASS_MAPPING_PATH = DATA_DIR / "msl_synset_words-indexed.txt"
README_PATH = DATA_DIR / "README.txt"

# Project module and artifact directories
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
CONFUSION_MATRICES_DIR = EVALUATION_DIR / "confusion_matrices"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Standardized model-wise results directories
RESULTS_KNN_DIR = RESULTS_DIR / "01_KNN"
RESULTS_NAIVE_BAYES_DIR = RESULTS_DIR / "02_Naive_Bayes"
RESULTS_DECISION_TREE_DIR = RESULTS_DIR / "03_Decision_Tree"
RESULTS_RANDOM_FOREST_DIR = RESULTS_DIR / "04_Random_Forest"
RESULTS_SVM_DIR = RESULTS_DIR / "05_SVM"
RESULTS_MRSCATT_DIR = RESULTS_DIR / "06_MRSCAtt"
RESULTS_VIT_DIR = RESULTS_DIR / "07_ViT_B16"
RESULTS_RESNET50_DIR = RESULTS_DIR / "08_ResNet50"
RESULTS_EFFICIENTNET_B3_DIR = RESULTS_DIR / "09_EfficientNet_B3"
RESULTS_OVERALL_DIR = RESULTS_DIR / "10_Overall_Comparison"
RESULTS_UNCLASSIFIED_DIR = RESULTS_DIR / "99_Unclassified"

# Deep learning checkpoint directories
MODELS_RESNET50_DIR = SAVED_MODELS_DIR / "deep_learning" / "resnet50"
MODELS_MRSCATT_DIR = SAVED_MODELS_DIR / "deep_learning" / "mrscatt"
MODELS_VIT_DIR = SAVED_MODELS_DIR / "deep_learning" / "vit"
MODELS_EFFICIENTNET_B3_DIR = SAVED_MODELS_DIR / "deep_learning" / "efficientnet_b3"

# Compatibility aliases
DEEP_LEARNING_DIR = PROJECT_ROOT / "deep_learning"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"
PLOTS_DIR = RESULTS_UNCLASSIFIED_DIR / "eda_plots"

# Phase 5A (ResNet-50) compatibility
PHASE5A_RESULTS_DIR = RESULTS_RESNET50_DIR
PHASE5A_REPORTS_DIR = PHASE5A_RESULTS_DIR / "classification_reports"
PHASE5A_CONFUSION_DIR = PHASE5A_RESULTS_DIR / "confusion_matrices"
PHASE5A_PLOTS_DIR = PHASE5A_RESULTS_DIR / "plots"
PHASE5A_HISTORIES_DIR = PHASE5A_RESULTS_DIR / "histories"
PHASE5A_METRICS_DIR = PHASE5A_RESULTS_DIR / "metrics"
PHASE5A_MODELS_DIR = MODELS_RESNET50_DIR

# Phase 5B (MRSCAtt) compatibility
PHASE5B_RESULTS_DIR = RESULTS_MRSCATT_DIR
PHASE5B_REPORTS_DIR = PHASE5B_RESULTS_DIR / "classification_reports"
PHASE5B_CONFUSION_DIR = PHASE5B_RESULTS_DIR / "confusion_matrices"
PHASE5B_PREDICTIONS_DIR = PHASE5B_RESULTS_DIR / "predictions"
PHASE5B_PLOTS_DIR = PHASE5B_RESULTS_DIR / "plots"
PHASE5B_LOGS_DIR = PHASE5B_RESULTS_DIR / "logs"
PHASE5B_METRICS_DIR = PHASE5B_RESULTS_DIR / "metrics"
PHASE5B_MODELS_DIR = MODELS_MRSCATT_DIR

# Phase 5C (ViT-B/16) compatibility
PHASE5C_RESULTS_DIR = RESULTS_VIT_DIR
PHASE5C_REPORTS_DIR = PHASE5C_RESULTS_DIR / "classification_reports"
PHASE5C_CONFUSION_DIR = PHASE5C_RESULTS_DIR / "confusion_matrices"
PHASE5C_PREDICTIONS_DIR = PHASE5C_RESULTS_DIR / "predictions"
PHASE5C_PLOTS_DIR = PHASE5C_RESULTS_DIR / "plots"
PHASE5C_LOGS_DIR = PHASE5C_RESULTS_DIR / "logs"
PHASE5C_METRICS_DIR = PHASE5C_RESULTS_DIR / "metrics"
PHASE5C_MODELS_DIR = MODELS_VIT_DIR

# Phase 5D (EfficientNet-B3) compatibility
PHASE5D_RESULTS_DIR = RESULTS_EFFICIENTNET_B3_DIR
PHASE5D_REPORTS_DIR = PHASE5D_RESULTS_DIR / "classification_reports"
PHASE5D_CONFUSION_DIR = PHASE5D_RESULTS_DIR / "confusion_matrices"
PHASE5D_PREDICTIONS_DIR = PHASE5D_RESULTS_DIR / "predictions"
PHASE5D_PLOTS_DIR = PHASE5D_RESULTS_DIR / "plots"
PHASE5D_LOGS_DIR = PHASE5D_RESULTS_DIR / "logs"
PHASE5D_METRICS_DIR = PHASE5D_RESULTS_DIR / "metrics"
PHASE5D_MODELS_DIR = MODELS_EFFICIENTNET_B3_DIR

# Dataset constants
NUM_EXPECTED_CLASSES = 25
CLASS_IDS_RANGE = range(0, 25)

# The one globally inactive NASA class: 0 samples in ALL official splits.
ZERO_INSTANCE_CLASS_ID = 22  # "sun"

# 24 active NASA class IDs (excludes class 22 "sun").
# Used by classical ML models (raw NASA IDs as labels).
ACTIVE_CLASS_IDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 23, 24
]
NUM_ACTIVE_CLASSES = 24

# NOTE: NASA IDs 5 ("drill holes") and 23 ("turret") are ACTIVE classes but
# have zero ground-truth samples in the official TEST split (1,305 images).
# They must NOT be removed from the Macro-F1 denominator.
# zero_division=0 assigns F1=0.0 to them; the average is over all 24 classes.
ZERO_SUPPORT_IN_TEST = [5, 23]  # Active but 0 test samples

# Standard image resolution (browse images)
STANDARD_IMAGE_WIDTH = 256
STANDARD_IMAGE_HEIGHT = 256
