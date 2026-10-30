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
from preprocess import MEAN, STD, bgr_to_tensor, letterbox, to_letterbox

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
    """Training convenience wrapper around the shared letterbox transform."""
    canvas, valid, transform = letterbox(bgr, size)
    mapped = [np.rint(to_letterbox(p, transform)).astype(np.int32)
              for p in polygons]
    return canvas, mapped, valid.astype(np.uint8)


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

        image_tensor = bgr_to_tensor(image)
        return {
            "image": image_tensor,  # [3,H,W] float32
            "kernels": torch.from_numpy(kernels.copy()).float(),  # [3,H,W]
            "training_mask": torch.from_numpy(valid.copy()).float(),  # [H,W]
            "name": name,
        }
