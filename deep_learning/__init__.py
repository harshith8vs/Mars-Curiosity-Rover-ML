"""Deep learning package for Mars rover image classification."""

from deep_learning.datasets import (
    MarsRoverDataset,
    load_synset_mapping,
    load_split_records,
    build_active_class_mappings,
    compute_training_class_weights
)
from deep_learning.transforms import (
    LetterboxResize,
    get_train_transforms,
    get_eval_transforms
)
from deep_learning.models.resnet50 import (
    build_resnet50_baseline,
    freeze_backbone,
    unfreeze_all,
    get_differential_param_groups
)
from deep_learning.evaluation import (
    evaluate_split,
    recompute_metrics_from_predictions
)
from deep_learning.utils import (
    seed_everything,
    get_device,
    ensure_phase5a_dirs,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_per_class_f1_barchart,
    save_classification_report
)

__all__ = [
    "MarsRoverDataset",
    "load_synset_mapping",
    "load_split_records",
    "build_active_class_mappings",
    "compute_training_class_weights",
    "LetterboxResize",
    "get_train_transforms",
    "get_eval_transforms",
    "build_resnet50_baseline",
    "freeze_backbone",
    "unfreeze_all",
    "get_differential_param_groups",
    "evaluate_split",
    "recompute_metrics_from_predictions",
    "seed_everything",
    "get_device",
    "ensure_phase5a_dirs",
    "plot_learning_curves",
    "plot_confusion_matrix",
    "plot_per_class_f1_barchart",
    "save_classification_report"
]
