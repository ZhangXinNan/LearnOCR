"""Shared, model-independent image preprocessing for text detectors.

PSENet and a future DBNet may use the same letterbox/coordinate mapping.
Keep polygon shrinking and model-specific label generation out of this module.
"""

import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(bgr, size):
    """Aspect-preserving resize to square size; return image, content mask, map.

    The transform is (scale_x, scale_y, pad_left, pad_top, orig_width, orig_height).
    Separate x/y scales account for rounded resized dimensions.
    """
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
    valid = np.zeros((size, size), dtype=bool)
    valid[top:top + new_h, left:left + new_w] = True
    transform = (new_w / w, new_h / h, left, top, w, h)
    return canvas, valid, transform


def to_letterbox(points, transform):
    """Map Nx2 coordinates from original pixels to letterbox pixels."""
    sx, sy, left, top, _, _ = transform
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[..., 0] = pts[..., 0] * sx + left
    pts[..., 1] = pts[..., 1] * sy + top
    return pts


def to_original(points, transform):
    """Map Nx2 letterbox coordinates back to original image pixels."""
    sx, sy, left, top, width, height = transform
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[..., 0] = np.clip((pts[..., 0] - left) / sx, 0, width - 1)
    pts[..., 1] = np.clip((pts[..., 1] - top) / sy, 0, height - 1)
    return pts


def bgr_to_tensor(bgr, batched=False):
    """BGR uint8 -> ImageNet-normalized RGB float32 tensor, CHW or 1CHW."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy())
    return tensor.unsqueeze(0) if batched else tensor
