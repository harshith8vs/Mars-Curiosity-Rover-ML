"""Decision Tree classifier implementation."""

from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def build_decision_tree_model(
    criterion: str = "gini",
    max_depth: Optional[int] = 15,
    min_samples_split: int = 2,
    class_weight: Optional[Union[str, Dict[int, float]]] = None,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """Construct a DecisionTreeClassifier with given hyperparameters."""
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state
    )


def train_decision_tree(
    model: DecisionTreeClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> DecisionTreeClassifier:
    """Trains DecisionTreeClassifier on feature matrix X_train and labels y_train."""
    model.fit(X_train, y_train)
    return model


def predict_decision_tree(
    model: DecisionTreeClassifier,
    X: np.ndarray
) -> np.ndarray:
    """Generates class predictions using trained DecisionTreeClassifier."""
    return model.predict(X)
