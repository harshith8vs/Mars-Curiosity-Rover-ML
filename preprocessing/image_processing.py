"""Image preprocessing routines for loading, channel standardization, resizing, and normalization."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk and return as a NumPy ndarray."""
    path_obj = Path(image_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(path_obj) as pil_img:
            return np.array(pil_img)
    except Exception as e:
        raise ValueError(f"Failed to decode image {image_path}: {e}") from e


def standardize_channels(image: np.ndarray) -> np.ndarray:
    """Standardize input image array to 3-channel RGB (replicates grayscale, drops/blends alpha)."""
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # 2D grayscale image (H, W)
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)

    # 3D image array
    if image.ndim == 3:
        num_channels = image.shape[2]
        if num_channels == 1:
            # Single-channel 3D (H, W, 1) -> replicate to RGB
            return np.repeat(image, 3, axis=-1)
        elif num_channels == 3:
            # Already 3 channels
            return image
        elif num_channels == 4:
            # RGBA -> RGB (composite over black/neutral background)
            rgb = image[:, :, :3].astype(np.float32)
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            blended = rgb * alpha
            return np.clip(blended, 0, 255).astype(image.dtype)
        else:
            raise ValueError(f"Unsupported channel count: {num_channels} (shape={image.shape})")

    raise ValueError(f"Unsupported image array dimension: {image.ndim} (shape={image.shape})")


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    preserve_aspect_ratio: bool = True,
    pad_mode: str = "constant",
    pad_value: int = 0
) -> np.ndarray:
    """Resize image to target_size, optionally preserving aspect ratio via letterboxing."""
    target_w, target_h = target_size
    orig_h, orig_w = image.shape[:2]

    # Quick exit if image already matches target size
    if orig_w == target_w and orig_h == target_h:
        return image.copy()

    if not preserve_aspect_ratio:
        interp = cv2.INTER_AREA if (target_w < orig_w or target_h < orig_h) else cv2.INTER_LINEAR
        return cv2.resize(image, (target_w, target_h), interpolation=interp)

    # Aspect-ratio-preserving letterboxing
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Calculate padding amounts to center the image
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    if pad_mode == "constant":
        padded = cv2.copyMakeBorder(
            scaled_img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value)
        )
    elif pad_mode == "edge":
        padded = cv2.copyMakeBorder(
            scaled_img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
    elif pad_mode == "reflect":
        padded = cv2.copyMakeBorder(
            scaled_img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}. Choose from 'constant', 'edge', 'reflect'.")

    return padded


def normalize_pixels(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize pixel values linearly to float32 [0.0, 1.0]."""
    if method == "minmax":
        return image.astype(np.float32) / 255.0
    elif method == "none":
        return image.astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (256, 256),
    preserve_aspect_ratio: bool = True,
    pad_mode: str = "constant",
    pad_value: int = 0,
    normalize: bool = True
) -> np.ndarray:
    """Load, standardize channels, letterbox-resize, and normalize an image."""
    raw_img = load_image(image_path)
    rgb_img = standardize_channels(raw_img)
    resized_img = resize_image(
        rgb_img,
        target_size=target_size,
        preserve_aspect_ratio=preserve_aspect_ratio,
        pad_mode=pad_mode,
        pad_value=pad_value
    )
    if normalize:
        return normalize_pixels(resized_img, method="minmax")
    return resized_img
