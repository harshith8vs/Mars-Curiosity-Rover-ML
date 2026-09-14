"""Support Vector Machine (SVM) classifier implementation."""

from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.svm import SVC


def build_svm_model(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    class_weight: Optional[Union[str, Dict[int, float]]] = None,
    random_state: int = 42
) -> SVC:
    """Construct an SVC classifier with given hyperparameters."""
    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        class_weight=class_weight,
        random_state=random_state
    )


def train_svm(
    model: SVC,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> SVC:
    """Trains SVC classifier on feature matrix X_train and labels y_train."""
    model.fit(X_train, y_train)
    return model


def predict_svm(
    model: SVC,
    X: np.ndarray
) -> np.ndarray:
    """Generates class predictions using trained SVC classifier."""
    return model.predict(X)
