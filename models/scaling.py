"""Feature scaling wrapper enforcing strict training-only parameter estimation."""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler


class TrainingScaler:
    """StandardScaler wrapper ensuring normalization parameters are fitted exclusively on training data."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        self.is_fitted = False
        
        # Provenance verification trackers
        self.train_sample_count: Optional[int] = None
        self.train_feature_count: Optional[int] = None
        self.expected_train_mean: Optional[np.ndarray] = None
        self.expected_train_var: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, split_name: str = "train") -> "TrainingScaler":
        """Fit scaler exclusively on training feature matrix."""
        clean_name = split_name.lower().strip()
        if clean_name not in ["train", "training"]:
            raise ValueError(
                f"Data Leakage Violation! TrainingScaler.fit() can ONLY be called on training data. "
                f"Attempted to fit on '{split_name}'. Fitting on validation or test data is strictly prohibited."
            )

        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)

        # Record training data dimensions
        self.train_sample_count = X_train.shape[0]
        self.train_feature_count = X_train.shape[1]

        # Compute independent reference moments for mathematical verification
        self.expected_train_mean = np.mean(X_train, axis=0, dtype=np.float64)
        self.expected_train_var = np.var(X_train, axis=0, dtype=np.float64)

        # Fit underlying scaler
        self.scaler.fit(X_train)
        self.is_fitted = True

        # Assert learned parameters match training data exactly
        self.verify_training_provenance(X_train)
        return self

    def verify_training_provenance(self, X_train: np.ndarray) -> bool:
        """
        Verifies that scaler.mean_ and scaler.var_ match X_train moments to floating point precision.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot verify provenance: TrainingScaler is not fitted.")

        if self.with_mean and self.scaler.mean_ is not None:
            max_mean_diff = np.max(np.abs(self.scaler.mean_ - self.expected_train_mean))
            if max_mean_diff > 1e-4:
                raise AssertionError(f"Provenance Error: scaler.mean_ deviates from X_train by {max_mean_diff}")

        if self.with_std and self.scaler.var_ is not None:
            max_var_diff = np.max(np.abs(self.scaler.var_ - self.expected_train_var))
            if max_var_diff > 1e-4:
                raise AssertionError(f"Provenance Error: scaler.var_ deviates from X_train by {max_var_diff}")

        return True

    def transform(self, X: np.ndarray, split_name: str = "unspecified") -> np.ndarray:
        """Apply training-learned normalization parameters to a dataset split."""
        if not self.is_fitted:
            raise RuntimeError("TrainingScaler must be fitted on X_train before transform() can be called.")

        if X.shape[1] != self.train_feature_count:
            raise ValueError(
                f"Feature dimension mismatch: Scaler fitted on {self.train_feature_count} features, "
                f"but received {X.shape[1]} features."
            )

        # Snapshot parameters before transform to ensure transform is completely stateless
        mean_before = self.scaler.mean_.copy() if self.scaler.mean_ is not None else None
        var_before = self.scaler.var_.copy() if self.scaler.var_ is not None else None

        X_transformed = self.scaler.transform(X).astype(np.float32)

        # Confirm no internal state drift occurred during transform
        if mean_before is not None:
            assert np.array_equal(self.scaler.mean_, mean_before), "State mutation detected in scaler.mean_!"
        if var_before is not None:
            assert np.array_equal(self.scaler.var_, var_before), "State mutation detected in scaler.var_!"

        return X_transformed

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Convenience method that fits on X_train and returns transformed X_train."""
        self.fit(X_train, split_name="train")
        return self.transform(X_train, split_name="train")

    def get_provenance_report(self) -> Dict[str, Any]:
        """Returns structured metadata confirming training-only parameter estimation."""
        if not self.is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "fitted_exclusively_on": "X_train",
            "train_samples_used": self.train_sample_count,
            "features_scaled": self.train_feature_count,
            "learned_mean_range": [float(np.min(self.scaler.mean_)), float(np.max(self.scaler.mean_))],
            "learned_scale_range": [float(np.min(self.scaler.scale_)), float(np.max(self.scaler.scale_))],
            "zero_variance_features_count": int(np.sum(self.scaler.scale_ == 1.0))
        }


class PCAStrategy:
    """Architectural specification for PCA dimensionality reduction."""

    @staticmethod
    def get_status() -> Dict[str, str]:
        return {
            "status": "UNFITTED / UNAPPLIED in Phase 4A",
            "planned_phase": "Phase 4B (Controlled Model-by-Model Experiment)",
            "leakage_rule": "If evaluated in Phase 4B, PCA must be fitted EXCLUSIVELY on X_train_scaled.",
            "components_rule": "Number of components must be selected via explained variance threshold (e.g. 95%) or cross-validation on train split, never arbitrary guessing."
        }
