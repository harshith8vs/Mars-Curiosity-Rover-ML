"""K-Nearest Neighbors (KNN) classifier implementation."""

from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def build_knn_model(
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "euclidean",
    n_jobs: int = -1
) -> KNeighborsClassifier:
    """Construct a KNeighborsClassifier with given hyperparameters."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=n_jobs
    )


def train_knn(
    model: KNeighborsClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> KNeighborsClassifier:
    """Trains the KNN classifier on feature matrix X_train and labels y_train."""
    model.fit(X_train, y_train)
    return model


def predict_knn(
    model: KNeighborsClassifier,
    X: np.ndarray
) -> np.ndarray:
    """Generates class predictions using trained KNN classifier."""
    return model.predict(X)
