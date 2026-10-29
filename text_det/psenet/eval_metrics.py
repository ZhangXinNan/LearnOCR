"""Step 7: educational ICDAR2015 polygon-level detection metrics.

Uses convex quadrilaterals + OpenCV only.  Similar to ICDAR15 IoU matching,
NOT a drop-in replacement for the competition's official RRC evaluator.
"""
from pathlib import Path

import cv2
import numpy as np


def read_icdar_annotations(path):
    """Read original-pixel GT quadrilaterals and ### flags without pyclipper."""
    polygons, ignored = [], []
    with Path(path).open('r', encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip('\r\n').split(',', 8)
            if len(fields) != 9:
                raise ValueError(f'Invalid ICDAR annotation in {path}: {line!r}')
            polygons.append(np.asarray([float(x) for x in fields[:8]],
                                       dtype=np.float32).reshape(4, 2))
            ignored.append(fields[8].strip() == '###')
    return polygons, ignored


def _convex_polygon(points):
    """Ensure order-independent convex polygon, or return None if degenerate."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) < 3 or abs(cv2.contourArea(hull)) < 1e-6:
        return None
    return np.ascontiguousarray(hull, dtype=np.float32)


def polygon_area(polygon):
    poly = _convex_polygon(polygon)
    return 0.0 if poly is None else abs(float(cv2.contourArea(poly)))


def polygon_intersection(a, b):
    p, q = _convex_polygon(a), _convex_polygon(b)
    if p is None or q is None:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(p, q)
    return max(0.0, float(intersection))


def polygon_iou(a, b):
    intersection = polygon_intersection(a, b)
    union = polygon_area(a) + polygon_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def counts_to_metrics(tp, fp, fn):
    """Micro metrics. No positives anywhere -> report zeros, not a fake 100%."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_image(gt_polygons, gt_ignored, detections,
                   iou_threshold=0.5, ignore_overlap=0.5):
    """One-to-one matching; greedy highest-IoU pairs first.

    `detections` entries are {'box_xy': [[x,y], ...], 'score': ...}.
    Detection suppressed when over 50% of its OWN area overlaps any ### GT.
    Then match remaining detections to care GT at IoU > 0.5.
    """
    if len(gt_polygons) != len(gt_ignored):
        raise ValueError('GT polygon and ignore-flag lengths differ')
    if not (0 < iou_threshold < 1 and 0 < ignore_overlap < 1):
        raise ValueError('thresholds must be between 0 and 1')
    ignored_gt = [i for i, flag in enumerate(gt_ignored) if flag]
    care_gt = [i for i, flag in enumerate(gt_ignored) if not flag]
    ignored_det = set()
    for j, det in enumerate(detections):
        box = det['box_xy']
        area = polygon_area(box)
        if area > 0 and any(
            polygon_intersection(box, gt_polygons[i]) / area > ignore_overlap
            for i in ignored_gt
        ):
            ignored_det.add(j)

    candidates = []
    for i in care_gt:
        for j, det in enumerate(detections):
            if j in ignored_det:
                continue
            overlap = polygon_iou(gt_polygons[i], det['box_xy'])
            if overlap > iou_threshold:  # ICDAR: strictly greater than 0.5
                candidates.append((overlap, i, j))

    candidates.sort(reverse=True)  # educational: greedy best-IoU matching
    matched_gt, matched_det, pairs = set(), set(), []
    for overlap, i, j in candidates:
        if i not in matched_gt and j not in matched_det:
            matched_gt.add(i)
            matched_det.add(j)
            pairs.append({'gt_index': i, 'det_index': j, 'iou': round(overlap, 6)})

    false_positive = [j for j in range(len(detections))
                      if j not in ignored_det and j not in matched_det]
    false_negative = [i for i in care_gt if i not in matched_gt]
    tp, fp, fn = len(pairs), len(false_positive), len(false_negative)
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'num_care_gt': len(care_gt),
        'num_care_det': len(detections) - len(ignored_det),
        'pairs': pairs,
        'matched_gt': sorted(matched_gt),
        'matched_det': sorted(matched_det),
        'ignored_gt': ignored_gt,
        'ignored_det': sorted(ignored_det),
        'false_positive_det': false_positive,
        'false_negative_gt': false_negative,
        **counts_to_metrics(tp, fp, fn),
    }
