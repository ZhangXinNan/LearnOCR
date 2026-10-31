"""Teaching implementation of Progressive Scale Expansion (PSE).

Input channel order: K1 (smallest), K2, ..., K_last (complete text).
No PyTorch and no compiled PSE extension are required.
"""
from collections import deque

import cv2
import numpy as np


# Exact neighbors for FOUR-connectivity (also used for the seed components).
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def binarize_kernels(probabilities, threshold=0.5):
    """[K,H,W] sigmoid probabilities -> nested bool masks, small to large.

    Predicted channels may NOT be nested, even though training GT is nested.
    Intersect each smaller mask with the next larger one to make nesting valid.
    """
    probs = np.asarray(probabilities)
    if probs.ndim != 3 or probs.shape[0] < 2:
        raise ValueError("Expected kernel probabilities of shape [K,H,W], K>=2")
    if not np.isfinite(probs).all():
        raise ValueError("Kernel probabilities must be finite")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be strictly between 0 and 1")

    masks = np.ascontiguousarray(probs >= threshold)
    for k in range(masks.shape[0] - 2, -1, -1):
        masks[k] &= masks[k + 1]
    return masks


def pse_expand(masks, min_kernel_area=3, return_steps=False):
    """Recover instance IDs with seed components and BFS scale expansion.

    Args:
      masks: bool or 0/1 array [K,H,W], ordered smallest -> largest.
      min_kernel_area: reject tiny connected seed blobs (pixels at INPUT size).
      return_steps: additionally return a COPY of label image at every scale.

    Returns:
      labels: int32 [H,W], 0=background, positive integer=instance ID.
      If return_steps, also returns a list of K int32 label images.

    Teaching choice: use 4-connectivity consistently for components and BFS.
    Conflicts at a shared pixel are resolved by the first BFS arrival.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[0] < 2:
        raise ValueError("Expected binary masks [K,H,W] with at least 2 scales")
    if min_kernel_area < 1:
        raise ValueError("min_kernel_area must be >= 1")
    masks = np.ascontiguousarray(masks > 0)
    # Protect the algorithm even when manually supplied masks are not nested.
    for k in range(masks.shape[0] - 2, -1, -1):
        masks[k] &= masks[k + 1]

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        masks[0].astype(np.uint8), connectivity=4
    )
    labels = labels.astype(np.int32, copy=False)
    # Remove noise *before* growing: a stray seed must not get a full region.
    keep = stats[:, cv2.CC_STAT_AREA] >= min_kernel_area
    keep[0] = False  # component 0 = background, never a seed
    labels[~keep[labels]] = 0

    steps = [labels.copy()] if return_steps else None
    height, width = labels.shape

    for scale in range(1, masks.shape[0]):
        allowed = masks[scale]
        # Every labeled pixel is a valid BFS starting point for this scale.
        # All seeds enter before any newly expanded pixels, making growth fair.
        ys, xs = np.nonzero(labels)
        queue = deque(zip(ys.tolist(), xs.tolist()))
        while queue:
            y, x = queue.popleft()
            instance_id = labels[y, x]
            for dy, dx in NEIGHBORS:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width
                        and allowed[ny, nx] and labels[ny, nx] == 0):
                    labels[ny, nx] = instance_id
                    queue.append((ny, nx))
        if return_steps:
            steps.append(labels.copy())

    if return_steps:
        return labels, steps
    return labels


def extract_instances(labels, text_probability, min_area=30, min_score=0.5):
    """Extract one quadrilateral and optional full contour per final instance.

    These coordinates are in the mask coordinate system; when inference uses
    letterbox, predict.py maps them back to the original image.
    """
    labels = np.asarray(labels)
    score_map = np.asarray(text_probability)
    if labels.ndim != 2 or score_map.shape != labels.shape:
        raise ValueError("Need labels [H,W] and K_last probabilities [H,W]")
    if min_area < 1 or not (0.0 <= min_score <= 1.0):
        raise ValueError("min_area must be >=1 and min_score in [0,1]")

    instances = []
    for instance_id in np.unique(labels):
        if instance_id == 0:
            continue
        component = labels == instance_id
        area = int(component.sum())
        if area < min_area:
            continue
        score = float(score_map[component].mean())
        if score < min_score:
            continue

        contours, _ = cv2.findContours(
            (component.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)[:, 0, :]  # [N,2]
        if len(contour) < 3:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        instances.append({
            "id": int(instance_id), "area": area, "score": score,
            "box": box, "contour": contour.astype(np.float32),
        })
    return instances


def instance_color(instance_id):
    """Deterministic BGR color for an instance."""
    n = int(instance_id)
    return 60 + (67 * n) % 196, 60 + (113 * n) % 196, 60 + (157 * n) % 196


def colorize_labels(labels):
    """Visualize integer instance IDs with reproducible colors (BGR)."""
    labels = np.asarray(labels)
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(labels):
        if instance_id == 0:
            continue
        color = instance_color(instance_id)
        image[labels == instance_id] = color
    return image
