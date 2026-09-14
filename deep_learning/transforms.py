"""Image preprocessing and augmentation transforms for deep learning models."""

from typing import Tuple
from PIL import Image
import torchvision.transforms as T

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LetterboxResize:
    """Aspect-ratio-preserving resize with neutral black letterbox padding."""
    def __init__(self, target_size: Tuple[int, int] = (224, 224), pad_color: Tuple[int, int, int] = (0, 0, 0)):
        self.target_w, self.target_h = target_size
        self.pad_color = pad_color

    def __call__(self, img: Image.Image) -> Image.Image:
        orig_w, orig_h = img.size
        
        # Scale proportionally to fit within target bounding box
        scale = min(self.target_w / orig_w, self.target_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        
        resized_img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.target_w, self.target_h), self.pad_color)
        
        # Center image within padding canvas
        pad_x = (self.target_w - new_w) // 2
        pad_y = (self.target_h - new_h) // 2
        canvas.paste(resized_img, (pad_x, pad_y))
        
        return canvas

    def __repr__(self) -> str:
        return f"LetterboxResize(target_size=({self.target_w}, {self.target_h}), pad_color={self.pad_color})"


def get_train_transforms(target_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """Training data augmentation pipeline (letterbox resize, crop, rotation, jitter, flip)."""
    initial_scale = int(round(target_size[0] * 256.0 / 224.0))
    return T.Compose([
        LetterboxResize(target_size=(initial_scale, initial_scale)),
        T.RandomResizedCrop(
            size=target_size,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.RandomRotation(degrees=(-10, 10)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_eval_transforms(target_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """Deterministic validation and test preprocessing pipeline (letterbox resize + normalization)."""
    return T.Compose([
        LetterboxResize(target_size=target_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
