"""Step 5 independent self-test: creates temporary synthetic ICDAR-like data.

Requires Step 2/3/4 libraries including pyclipper, no ICDAR download needed.
Run: python text_detection/psenet/test_training.py
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from dataset import ICDAR2015Dataset, letterbox_and_polygons
from loss import PSENetLoss
from model import MiniPSENet
from train import pixel_text_dice


def main():
    original = np.zeros((80, 160, 3), np.uint8)
    polygon = np.array([[20, 20], [130, 20], [130, 45], [20, 45]])
    canvas, scaled, content = letterbox_and_polygons(original, [polygon], 192)
    assert canvas.shape == (192, 192, 3)
    assert scaled[0].shape == (4, 2)
    assert content[0, 0] == 0 and content[95, 95] == 1
    print("Letterbox and polygons: PASS")

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train_images").mkdir()
        (root / "train_gt").mkdir()
        cv2.rectangle(original, (20, 20), (130, 45), (255, 255, 255), -1)
        cv2.imwrite(str(root / "train_images" / "img_1.jpg"), original)
        (root / "train_gt" / "gt_img_1.txt").write_text(
            "20,20,130,20,130,45,20,45,HELLO\n"
            "10,55,50,55,50,70,10,70,###\n", encoding="utf-8"
        )
        ds = ICDAR2015Dataset(root, ["img_1.jpg"], size=192, augment=False)
        sample = ds[0]
        assert sample["image"].shape == (3, 192, 192)
        assert sample["kernels"].shape == (3, 192, 192)
        assert sample["training_mask"].shape == (192, 192)
        k = sample["kernels"].numpy()
        assert k[0].sum() > 0 and k[0].sum() < k[1].sum() < k[2].sum()
        assert sample["training_mask"][0, 0] == 0  # padded region
        assert sample["training_mask"][155, 30] == 0  # ### ignore
        print("Dataset labels / padding / ignore mask: PASS")

        torch.manual_seed(42)
        model = MiniPSENet(num_kernels=3, pretrained=False)
        model.train()
        criterion = PSENetLoss(kernel_mask_mode="predicted")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        img = sample["image"].unsqueeze(0)
        gt = sample["kernels"].unsqueeze(0)
        mask = sample["training_mask"].unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        out = model(img)
        result = criterion(out, gt, mask)
        assert torch.isfinite(result["loss"]).item()
        result["loss"].backward()
        assert model.head[-1].weight.grad is not None
        optimizer.step()
        metric = pixel_text_dice(out, gt, mask)
        assert 0 <= metric <= 1
        print(f"Full training step: PASS, loss={result['loss'].item():.4f}, pixel_Dice={metric:.4f}")

    print("ALL STEP 5 CHECKS PASSED")


if __name__ == "__main__":
    main()
