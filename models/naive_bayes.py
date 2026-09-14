"""Gaussian Naive Bayes classifier implementation."""

from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.naive_bayes import GaussianNB


def build_naive_bayes_model(
    priors: Optional[np.ndarray] = None,
    var_smoothing: float = 1e-9
) -> GaussianNB:
    """Construct a GaussianNB classifier with given priors and smoothing."""
    return GaussianNB(
        priors=priors,
        var_smoothing=var_smoothing
    )


def train_naive_bayes(
    model: GaussianNB,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> GaussianNB:
    """Trains GaussianNB classifier on feature matrix X_train and labels y_train."""
    model.fit(X_train, y_train)
    return model


def predict_naive_bayes(
    model: GaussianNB,
    X: np.ndarray
) -> np.ndarray:
    """Generates class predictions using trained GaussianNB classifier."""
    return model.predict(X)
