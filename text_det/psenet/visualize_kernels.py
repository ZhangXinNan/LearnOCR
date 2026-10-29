"""PSENet Step 2: visualize multi-scale labels for an ICDAR 2015 image."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from kernel import RATIOS, generate_kernel_labels, read_icdar_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/icdar2015")
    parser.add_argument("--name", default="img_1.jpg")
    parser.add_argument("--output", default="outputs/psenet_step2/img_1.png")
    args = parser.parse_args()

    data = Path(args.data)
    image_path = data / "train_images" / args.name
    gt_path = data / "train_gt" / f"gt_{Path(args.name).stem}.txt"
    image = cv2.imread(
        str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        raise FileNotFoundError(image_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    polygons, ignored = read_icdar_gt(gt_path)
    kernels, training_mask = generate_kernel_labels(image.shape, polygons, ignored)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compare the original image with all three binary targets and the ignore mask.
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    for i, (ratio, kernel) in enumerate(zip(RATIOS, kernels), start=1):
        axes[i].imshow(kernel, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"K{i}: r={ratio:.2f}")
    axes[4].imshow(training_mask, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("Training mask (black=ignore)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)

    counts = kernels.sum(axis=(1, 2)).tolist()
    assert np.all(kernels[0] <= kernels[1])
    assert np.all(kernels[1] <= kernels[2])
    assert counts[0] <= counts[1] <= counts[2], counts
    print(f"Image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    print(f"Instances: {len(polygons)}, ignored: {sum(ignored)}")
    print(f"Positive pixels [K1,K2,K3]: {counts}")
    print(f"Training mask ignored pixels: {int((training_mask == 0).sum())}")
    print(f"Nested mask checks: PASS")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
