"""Tests for PSE (no trained model or ICDAR dataset required)."""
import numpy as np

from pse import binarize_kernels, extract_instances, pse_expand


def overlapping_text_example():
    """Two separate K1 seeds, but K2 and K3 become ONE connected region."""
    masks = np.zeros((3, 56, 100), np.uint8)
    # K1: two separate small cores
    masks[0, 17:24, 9:34] = 1
    masks[0, 30:37, 57:82] = 1
    # K2: wider regions touch via a narrow bridge
    masks[1, 13:27, 5:44] = 1
    masks[1, 26:41, 48:88] = 1
    masks[1, 24:30, 40:51] = 1  # bridge -> one connected K2
    # K3: complete text candidates are touching
    masks[2, 8:30, 3:48] = 1
    masks[2, 24:46, 43:93] = 1  # K3 overlaps
    # Ensure K1, K2 nested in K3 (test geometry intentionally overlaps)
    masks[1] &= masks[2]
    masks[0] &= masks[1]
    return masks


def run_tests():
    masks = overlapping_text_example()
    labels, steps = pse_expand(masks, min_kernel_area=3, return_steps=True)
    assert len(steps) == 3
    ids = np.unique(labels[labels > 0]).tolist()
    assert len(ids) == 2, f"Expected two instances, got {ids}"
    assert labels[20, 20] != labels[34, 70]  # distinct text instances
    assert labels[20, 20] > 0 and labels[34, 70] > 0
    assert np.count_nonzero(steps[0]) < np.count_nonzero(steps[1]) < np.count_nonzero(steps[2])
    assert np.all(labels[masks[-1] == 0] == 0)
    # Once a pixel receives an ID, later scales must not reassign it.
    assert np.all(steps[0][steps[0] > 0] == steps[2][steps[0] > 0])
    assert np.all(steps[1][steps[1] > 0] == steps[2][steps[1] > 0])
    print("Two touching text regions stay separate: PASS")

    probs = masks.astype(np.float32) * 0.9 + 0.05
    # Artificially create a stray small kernel outside K3: it must disappear.
    probs[0, 0, 0] = 0.99
    nested = binarize_kernels(probs)
    assert not nested[0, 0, 0]
    assert np.all(nested[0] <= nested[1])
    assert np.all(nested[1] <= nested[2])
    assert len(np.unique(pse_expand(nested))[1:]) == 2
    print("Non-nested predictions are safely intersected: PASS")

    filtered = masks.copy()
    filtered[0, 9, 90] = 1
    filtered[1, 9, 90] = 1
    filtered[2, 9:12, 89:93] = 1
    labels2 = pse_expand(filtered, min_kernel_area=2)
    assert labels2[9, 90] == 0  # one-pixel seed was rejected
    assert len(np.unique(labels2[labels2 > 0])) == 2
    print("Tiny false-positive seeds do not grow: PASS")

    score = np.ones(labels.shape, np.float32) * 0.95
    detections = extract_instances(labels, score, min_area=50, min_score=0.5)
    assert len(detections) == 2
    assert all(d['box'].shape == (4, 2) for d in detections)
    score[labels == detections[0]['id']] = 0.1
    assert len(extract_instances(labels, score, min_area=50, min_score=0.5)) == 1
    print("Box extraction and confidence filtering: PASS")

    empty = np.zeros((3, 25, 25), np.uint8)
    assert not pse_expand(empty).any()
    assert not extract_instances(np.zeros((25, 25), np.int32),
                                 np.zeros((25, 25), np.float32))
    print("Empty predictions: PASS")
    print("ALL STEP 6 TESTS PASSED")


if __name__ == '__main__':
    run_tests()
