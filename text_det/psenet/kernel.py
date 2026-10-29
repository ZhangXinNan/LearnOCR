"""PSENet Step 2: generate multi-scale binary kernel labels from polygons."""

from pathlib import Path

import cv2
import numpy as np
import pyclipper

RATIOS = (0.50, 0.75, 1.00)  # smallest -> full text


def read_icdar_gt(path):
    """Return polygons (each 4x2) and ignore flags (###)."""
    polygons, ignored = [], []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.strip().split(",", 8)
            if len(fields) < 9:
                raise ValueError(f"Invalid ICDAR label: {line!r}")
            polygon = np.array([float(x) for x in fields[:8]])
            polygon = np.round(polygon).astype(np.int32).reshape(4, 2)
            polygons.append(polygon)
            ignored.append(fields[8].strip() == "###")
    return polygons, ignored


def shrink_polygon(polygon, ratio):
    """Inset a polygon using the PSENet area/perimeter offset formula."""
    polygon = np.asarray(polygon, dtype=np.int32)
    if ratio == 1.0:
        return [polygon]
    # 第一部分：计算内缩距离
    area = abs(cv2.contourArea(polygon))
    perimeter = cv2.arcLength(polygon, True)
    if area <= 0 or perimeter <= 0:
        return []

    distance = area * (1.0 - ratio**2) / perimeter
    # 第二部分：缩小多边形
    offset = pyclipper.PyclipperOffset()
    try:
        offset.AddPath(polygon.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 注意 -distance，负号表示向内部偏移。Execute() 返回的可能是多个 Polygon，因为不规则文本区域收缩后可能分裂成几块。
        paths = offset.Execute(-distance)
    except pyclipper.ClipperException:
        return []

    result = [np.asarray(path, dtype=np.int32) for path in paths if len(path) >= 3]
    return [path for path in result if abs(cv2.contourArea(path)) >= 1]


def generate_kernel_labels(image_shape, polygons, ignored, ratios=RATIOS):
    """Return kernels (N,H,W) and training_mask (H,W), all uint8 0/1."""
    h, w = image_shape[:2]
    ratios = tuple(ratios)
    if not ratios or ratios[-1] != 1.0 or any(
        not (0 < a < b <= 1) for a, b in zip(ratios, ratios[1:])
    ):
        raise ValueError("ratios must be increasing and end at 1.0")
    if len(polygons) != len(ignored):
        raise ValueError("polygons and ignored lengths differ")

    kernels = np.zeros((len(ratios), h, w), dtype=np.uint8)
    training_mask = np.ones((h, w), dtype=np.uint8)

    for polygon, is_ignored in zip(polygons, ignored):
        polygon = np.asarray(polygon, dtype=np.int32)
        if is_ignored:
            cv2.fillPoly(training_mask, [polygon], 0)
            continue
        # 第三部分：生成多个二值标签
        all_paths = [shrink_polygon(polygon, r) for r in ratios]
        if any(len(paths) == 0 for paths in all_paths):
            # No smallest seed => cannot supervise this instance properly.
            cv2.fillPoly(training_mask, [polygon], 0)
            continue

        for kernel, paths in zip(kernels, all_paths):
            cv2.fillPoly(kernel, paths, 1)

    # Remove any one-pixel violations caused by polygon rasterization.
    for i in range(len(ratios) - 2, -1, -1):
        kernels[i] &= kernels[i + 1]
    return kernels, training_mask


if __name__ == "__main__":
    # Independent geometric sanity test; no ICDAR files needed.
    box = np.array([[20, 20], [220, 20], [220, 60], [20, 60]], np.int32)
    ignore_box = np.array([[20, 80], [60, 80], [60, 110], [20, 110]], np.int32)
    k, mask = generate_kernel_labels((130, 250), [box, ignore_box], [False, True])
    counts = k.sum(axis=(1, 2)).tolist()
    assert 0 < counts[0] < counts[1] < counts[2], counts
    assert np.all(k[0] <= k[1]) and np.all(k[1] <= k[2])
    assert mask[95, 40] == 0 and mask[40, 100] == 1
    print("Synthetic test PASS; kernel pixel counts:", counts)
