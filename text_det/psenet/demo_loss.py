"""One real ICDAR 2015 image -> labels -> model -> loss -> backward.

This is a smoke test, NOT a training pipeline. From the LearnOCR root:
    python text_detection/psenet/demo_loss.py --name img_1.jpg --size 256
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from kernel import RATIOS, generate_kernel_labels, read_icdar_gt
from loss import PSENetLoss
from model import MiniPSENet


def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/icdar2015")
    parser.add_argument("--name", default="img_1.jpg")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    parser.add_argument("--kernel-mask-mode", choices=["predicted", "gt"],
                        default="predicted")
    args = parser.parse_args()
    if args.size < 64:
        raise ValueError("--size must be >=64")

    root = Path(args.data)
    image_file = root / "train_images" / args.name
    gt_file = root / "train_gt" / f"gt_{Path(args.name).stem}.txt"
    if not gt_file.is_file():
        raise FileNotFoundError(gt_file)
    bgr = cv2.imread(str(image_file),
                     cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise FileNotFoundError(image_file)

    polygons, ignored = read_icdar_gt(gt_file)
    kernels, valid_mask = generate_kernel_labels(bgr.shape, polygons, ignored,
                                                  ratios=RATIOS)
    if kernels.shape[0] != 3:
        raise ValueError("Expected three kernels: K1, K2, K3")

    # Resize images and masks together; nearest-neighbor preserves binary labels.
    # This square distortion is ONLY for Step 4 verification, not Step 5 training.
    new_size = (args.size, args.size)
    rgb = cv2.cvtColor(cv2.resize(bgr, new_size), cv2.COLOR_BGR2RGB)
    resized_kernels = np.stack([cv2.resize(k, new_size,
                                          interpolation=cv2.INTER_NEAREST)
                                for k in kernels])
    resized_valid = cv2.resize(valid_mask, new_size,
                               interpolation=cv2.INTER_NEAREST)

    device = get_device(args.device)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = ((image - mean) / std).unsqueeze(0).to(device)
    targets = torch.from_numpy(resized_kernels.copy()).float().unsqueeze(0).to(device)
    mask = torch.from_numpy(resized_valid.copy()).float().unsqueeze(0).to(device)

    torch.manual_seed(42)
    model = MiniPSENet(num_kernels=3, pretrained=False).to(device)
    model.train()
    loss_fn = PSENetLoss(kernel_mask_mode=args.kernel_mask_mode)
    model.zero_grad(set_to_none=True)
    logits = model(image)
    losses = loss_fn(logits, targets, mask)
    losses["loss"].backward()

    print(f"Device: {device}")
    print(f"Source: {image_file} / {gt_file}")
    print(f"Original image: {bgr.shape[1]}x{bgr.shape[0]}, boxes: {len(polygons)}")
    print(f"Image:   {tuple(image.shape)}")
    print(f"Targets: {tuple(targets.shape)}  (K1,K2,K3)")
    print(f"Mask:    {tuple(mask.shape)}  (valid=1, ignore=0)")
    print(f"Logits:  {tuple(logits.shape)}")
    print(f"Loss:    {losses['loss'].item():.6f}")
    print(f"Text:    {losses['loss_text'].item():.6f}")
    print(f"Kernel:  {losses['loss_kernel'].item():.6f}")
    for name, layer in (("Backbone", model.stem[0]),
                        ("FPN", model.lat2),
                        ("Head", model.head[-1])):
        grad = layer.weight.grad
        assert grad is not None and grad.isfinite().all(), name
        print(f"{name} grad norm: {grad.norm().item():.6f}")
    print("REAL IMAGE FORWARD + BACKWARD PASSED")


if __name__ == "__main__":
    main()
