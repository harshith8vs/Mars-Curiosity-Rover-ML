"""Preprocessing package for image operations, feature extraction, and EDA."""

from preprocessing.image_processing import (
    load_image,
    standardize_channels,
    resize_image,
    normalize_pixels,
    preprocess_image
)
from preprocessing.feature_extraction import (
    extract_features_from_preprocessed_image,
    get_feature_names,
    get_feature_group_breakdown
)
from preprocessing.eda import audit_class_distribution, audit_image_dimensions

__all__ = [
    "load_image",
    "standardize_channels",
    "resize_image",
    "normalize_pixels",
    "preprocess_image",
    "extract_features_from_preprocessed_image",
    "get_feature_names",
    "get_feature_group_breakdown",
    "audit_class_distribution",
    "audit_image_dimensions"
]

