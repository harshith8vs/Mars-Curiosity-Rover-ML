"""Random Forest classifier implementation."""

from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_random_forest_model(
    n_estimators: int = 100,
    max_depth: Optional[int] = 15,
    min_samples_split: int = 2,
    class_weight: Optional[Union[str, Dict[int, float]]] = None,
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """Construct a RandomForestClassifier with given hyperparameters."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )


def train_random_forest(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> RandomForestClassifier:
    """Trains RandomForestClassifier on feature matrix X_train and labels y_train."""
    model.fit(X_train, y_train)
    return model


def predict_random_forest(
    model: RandomForestClassifier,
    X: np.ndarray
) -> np.ndarray:
    """Generates class predictions using trained RandomForestClassifier."""
    return model.predict(X)
