"""Handcrafted feature extraction routines (color, texture, HOG, edges, gradients)."""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import cv2
from skimage.feature import hog, graycomatrix, graycoprops

from preprocessing.image_processing import preprocess_image


def rgb_to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert RGB float32 image to grayscale luminance using ITU-R BT.601 coefficients."""
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


# Group A: Color Statistics (19 features)
def extract_color_statistics(image: np.ndarray, luminance: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """Extract per-channel RGB moments (mean, std, min, max, median) and luminance stats."""
    if luminance is None:
        luminance = rgb_to_luminance(image)

    names: List[str] = []
    values: List[float] = []

    # Channel stats (R, G, B)
    channel_labels = ["R", "G", "B"]
    for i, ch_name in enumerate(channel_labels):
        ch = image[:, :, i]
        values.extend([
            float(np.mean(ch)),
            float(np.std(ch)),
            float(np.min(ch)),
            float(np.max(ch)),
            float(np.median(ch))
        ])
        names.extend([
            f"{ch_name}_mean",
            f"{ch_name}_std",
            f"{ch_name}_min",
            f"{ch_name}_max",
            f"{ch_name}_median"
        ])

    # Luminance / brightness stats
    values.extend([
        float(np.mean(luminance)),
        float(np.std(luminance)),
        float(np.min(luminance)),
        float(np.max(luminance))
    ])
    names.extend([
        "brightness_mean",
        "brightness_std",
        "brightness_min",
        "brightness_max"
    ])

    return np.array(values, dtype=np.float32), names


# Group B: Color Histograms (48 features)
def extract_color_histograms(image: np.ndarray, bins_per_channel: int = 16) -> Tuple[np.ndarray, List[str]]:
    """Extract normalized 1D histograms across R, G, B channels."""
    names: List[str] = []
    hist_list: List[np.ndarray] = []

    channel_labels = ["R", "G", "B"]
    for i, ch_name in enumerate(channel_labels):
        ch = image[:, :, i]
        hist, _ = np.histogram(ch, bins=bins_per_channel, range=(0.0, 1.0))
        total_pixels = hist.sum()
        normalized_hist = (hist / total_pixels) if total_pixels > 0 else np.zeros_like(hist, dtype=np.float32)
        hist_list.append(normalized_hist.astype(np.float32))
        names.extend([f"{ch_name}_hist_{b}" for b in range(bins_per_channel)])

    return np.concatenate(hist_list), names


# Group C: Texture Features (GLCM) (12 features)
def extract_glcm_texture(
    luminance: np.ndarray,
    distances: Tuple[int, ...] = (1,),
    angles: Tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
) -> Tuple[np.ndarray, List[str]]:
    """Extract Gray-Level Co-occurrence Matrix (GLCM) Haralick texture descriptors."""
    gray_uint8 = np.clip(luminance * 255.0, 0, 255).astype(np.uint8)
    glcm = graycomatrix(
        gray_uint8,
        distances=list(distances),
        angles=list(angles),
        levels=256,
        symmetric=True,
        normed=True
    )

    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    values: List[float] = []
    names: List[str] = []

    for prop in properties:
        prop_matrix = graycoprops(glcm, prop)  # shape (len(distances), len(angles))
        values.append(float(np.mean(prop_matrix)))
        values.append(float(np.std(prop_matrix)))
        names.append(f"glcm_{prop}_mean")
        names.append(f"glcm_{prop}_std")

    return np.array(values, dtype=np.float32), names


# ----------------------------------------------------------------------
# Feature Group D: HOG Features (8,100 features)
# ----------------------------------------------------------------------
def extract_hog_features(
    luminance: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2)
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract Histogram of Oriented Gradients (HOG) structural features.
    
    For a 256x256 image with (16, 16) cells and (2, 2) blocks:
    - (256/16 - 2 + 1) = 15 blocks along each axis -> 15 * 15 = 225 blocks.
    - 225 blocks * (2 * 2 cells/block) * 9 orientations = 8,100 features.
    
    Returns:
        Tuple of (feature_vector [8100], feature_names [8100]).
    """
    hog_feats = hog(
        luminance,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True
    ).astype(np.float32)

    names = [f"hog_{idx}" for idx in range(len(hog_feats))]
    return hog_feats, names


# Groups E & F: Edge and Gradient Features (2 Edge + 5 Gradient)
def extract_edge_and_gradient_features(luminance: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Extract Canny edge descriptors and Sobel gradient statistics."""
    # Sobel gradients
    sobelx = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Canny edges
    gray_uint8 = np.clip(luminance * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray_uint8, threshold1=50, threshold2=150)
    edge_pixels = (edges > 0)

    # Edge density
    edge_density = float(np.count_nonzero(edge_pixels) / edges.size)

    # Edge mean gradient (mean Sobel gradient magnitude at pixels detected by Canny)
    if np.any(edge_pixels):
        edge_mean_gradient = float(np.mean(grad_mag[edge_pixels]))
    else:
        edge_mean_gradient = 0.0

    edge_feats = np.array([edge_density, edge_mean_gradient], dtype=np.float32)
    edge_names = ["edge_density", "edge_mean_gradient"]

    # Gradient stats
    grad_feats = np.array([
        float(np.mean(grad_mag)),
        float(np.std(grad_mag)),
        float(np.min(grad_mag)),
        float(np.max(grad_mag)),
        float(np.percentile(grad_mag, 75))
    ], dtype=np.float32)
    grad_names = [
        "gradient_mean",
        "gradient_std",
        "gradient_min",
        "gradient_max",
        "gradient_p75"
    ]

    return edge_feats, edge_names, grad_feats, grad_names


# Unified feature extraction
def extract_features_from_preprocessed_image(
    image: np.ndarray,
    return_names: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """Extract concatenated 8,186-element feature vector from preprocessed image."""
    luminance = rgb_to_luminance(image)

    # Group A
    color_stats, names_a = extract_color_statistics(image, luminance=luminance)
    # Group B
    color_hists, names_b = extract_color_histograms(image, bins_per_channel=16)
    # Group C
    glcm_feats, names_c = extract_glcm_texture(luminance)
    # Group D
    hog_feats, names_d = extract_hog_features(luminance, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    # Groups E & F
    edge_feats, names_e, grad_feats, names_f = extract_edge_and_gradient_features(luminance)

    # Concatenate all feature arrays
    feature_vector = np.concatenate([
        color_stats,
        color_hists,
        glcm_feats,
        hog_feats,
        edge_feats,
        grad_feats
    ]).astype(np.float32)

    if return_names:
        all_names = names_a + names_b + names_c + names_d + names_e + names_f
        return feature_vector, all_names

    return feature_vector


def extract_image_features(
    image_path: Union[str, Path],
    return_names: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """Preprocess image from disk and extract full 8,186-element feature vector."""
    preprocessed_img = preprocess_image(
        image_path,
        target_size=(256, 256),
        preserve_aspect_ratio=True,
        pad_mode="constant",
        pad_value=0,
        normalize=True
    )
    return extract_features_from_preprocessed_image(preprocessed_img, return_names=return_names)


def get_feature_names() -> List[str]:
    """Return the ordered list of all 8,186 feature names."""
    dummy_img = np.zeros((256, 256, 3), dtype=np.float32)
    _, names = extract_features_from_preprocessed_image(dummy_img, return_names=True)
    return names


def get_feature_group_breakdown() -> Dict[str, int]:
    """Return feature counts grouped by descriptor category."""
    dummy_img = np.zeros((256, 256, 3), dtype=np.float32)
    lum = rgb_to_luminance(dummy_img)

    _, a = extract_color_statistics(dummy_img, lum)
    _, b = extract_color_histograms(dummy_img, 16)
    _, c = extract_glcm_texture(lum)
    _, d = extract_hog_features(lum)
    _, e, _, f = extract_edge_and_gradient_features(lum)

    counts = {
        "Group A - Color Statistics": len(a),
        "Group B - Color Histograms": len(b),
        "Group C - GLCM Texture": len(c),
        "Group D - HOG Structural": len(d),
        "Group E - Edge Descriptors": len(e),
        "Group F - Gradient Statistics": len(f),
    }
    counts["Total Features"] = sum(counts.values())
    return counts
