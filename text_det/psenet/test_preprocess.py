"""Test the shared preprocessing independently of ICDAR / pyclipper.

Run from the PSENet source directory: python test_preprocess.py
"""
import numpy as np
from preprocess import bgr_to_tensor, letterbox, to_letterbox, to_original
from predict import image_to_tensor, resize_letterbox


def main():
    for h, w, size in [(80, 160, 192), (720, 1280, 768),
                       (720, 1280, 1024), (333, 907, 736), (641, 479, 768)]:
        # Non-uniform synthetic pixels make image-resize regressions easier to catch.
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        canvas, valid, meta = letterbox(img, size)
        same_canvas, same_valid, same_meta = resize_letterbox(img, size)
        assert np.array_equal(canvas, same_canvas)
        assert np.array_equal(valid, same_valid)
        assert meta == same_meta
        assert canvas.shape == (size, size, 3)
        assert valid.any() and np.any(~valid)
        points = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
        recovered = to_original(to_letterbox(points, meta), meta)
        assert np.allclose(points, recovered, atol=1e-3)
        train_tensor = bgr_to_tensor(canvas)
        pred_tensor = image_to_tensor(canvas)
        assert pred_tensor.shape == (1, 3, size, size)
        assert np.array_equal(train_tensor.numpy(), pred_tensor[0].numpy())
    print("Preprocessing geometry / reverse mapping / normalization: PASS")


if __name__ == '__main__':
    main()
