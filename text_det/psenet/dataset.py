"""LearnOCR PSENet Step 5: ICDAR 2015 dataset with aspect-preserving letterbox.

Important: polygons are transformed FIRST; then Step 2 generates kernels in the
resized image coordinate system. Padding is ignored by the loss.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from kernel import RATIOS, generate_kernel_labels, read_icdar_gt

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_training_images(root):
    """Return sorted image basenames that have corresponding ICDAR GT files."""
    root = Path(root)
    images = root / "train_images"
    labels = root / "train_gt"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"Expected {images} and {labels} directories")
    names = sorted(p.name for p in images.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not names:
        raise ValueError(f"No image files found in {images}")
    missing = [name for name in names
               if not (labels / f"gt_{Path(name).stem}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing annotation files for {missing[:5]}")
    return names


def letterbox_and_polygons(bgr, polygons, size):
    """Resize keeping aspect ratio, center pad to size×size, transform polygons."""
    h, w = bgr.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("Invalid input image dimensions")
    if size < 64 or size % 32 != 0:
        raise ValueError("size must be >=64 and divisible by 32")

    scale = min(size / w, size / h)
    new_w = max(1, min(size, round(w * scale)))
    new_h = max(1, min(size, round(h * scale)))
    left = (size - new_w) // 2
    top = (size - new_h) // 2

    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    content_mask = np.zeros((size, size), dtype=np.uint8)
    content_mask[top:top + new_h, left:left + new_w] = 1

    # Actual x/y scales account for pixel rounding in new_w/new_h.
    sx, sy = new_w / w, new_h / h
    transformed = []
    for polygon in polygons:
        p = np.asarray(polygon, dtype=np.float32).copy()
        p[:, 0] = p[:, 0] * sx + left
        p[:, 1] = p[:, 1] * sy + top
        transformed.append(np.round(p).astype(np.int32))
    return canvas, transformed, content_mask


class ICDAR2015Dataset(Dataset):
    def __init__(self, root="data/icdar2015", names=None, size=512,
                 augment=False):
        self.root = Path(root)
        self.names = list(names) if names is not None else list_training_images(root)
        self.size = size
        self.augment = augment
        if size < 64 or size % 32 != 0:
            raise ValueError("size must be >=64 and divisible by 32")
        if not self.names:
            raise ValueError("Dataset has no images")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image_file = self.root / "train_images" / name
        gt_file = self.root / "train_gt" / f"gt_{Path(name).stem}.txt"
        image = cv2.imread(str(image_file),
                           cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(f"Could not read {image_file}")

        polygons, ignored = read_icdar_gt(gt_file)
        image, polygons, content_mask = letterbox_and_polygons(
            image, polygons, self.size
        )
        kernels, valid = generate_kernel_labels(
            image.shape, polygons, ignored, ratios=RATIOS
        )
        valid = (valid * content_mask).astype(np.uint8)

        # Mild photometric augmentation only; coordinates and masks stay aligned.
        # No flips: mirrored words would be unnatural training data.
        if self.augment:
            alpha = np.random.uniform(0.85, 1.15)  # contrast
            beta = np.random.uniform(-15, 15)     # brightness
            image = np.clip(image.astype(np.float32) * alpha + beta,
                            0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - MEAN) / STD
        image_tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy())
        return {
            "image": image_tensor,  # [3,H,W] float32
            "kernels": torch.from_numpy(kernels.copy()).float(),  # [3,H,W]
            "training_mask": torch.from_numpy(valid.copy()).float(),  # [H,W]
            "name": name,
        }
