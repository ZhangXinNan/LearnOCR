"""Step 3 smoke test: feature shapes, logits, probabilities and gradients.

Run from LearnOCR project root:
  python text_detection/psenet/test_model.py
  python text_detection/psenet/test_model.py --height 640 --width 640
"""

import argparse

import torch

from model import MiniPSENet


def choose_device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    args = parser.parse_args()
    assert args.height >= 32 and args.width >= 32, "Minimum input size is 32x32"

    torch.manual_seed(42)
    device = choose_device(args.device)
    model = MiniPSENet(num_kernels=3, pretrained=False).to(device)
    model.eval()
    x = torch.randn(1, 3, args.height, args.width, device=device)

    # Check full-resolution outputs and intermediate feature map sizes.
    with torch.no_grad():
        logits, features = model(x, return_features=True)
        probabilities = torch.sigmoid(logits)

    print(f"Device: {device}")
    print(f"Input: {tuple(x.shape)}")
    for name, feat in features.items():
        print(f"{name:12s} {tuple(feat.shape)}")
    print(f"Logits:       {tuple(logits.shape)}")
    print(f"Probabilities min/max: {probabilities.min().item():.4f} / "
          f"{probabilities.max().item():.4f}")

    assert logits.shape == (1, 3, args.height, args.width)
    assert torch.isfinite(logits).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert features["fused"].shape[1] == 256  # 4 pyramid levels x 64 channels
    assert features["small_logits"].shape[1] == 3

    # Check gradients can propagate through head, FPN, and ResNet backbone.
    # Keep this test small to work on CPU and Apple Silicon.
    model.train()
    model.zero_grad(set_to_none=True)
    small_x = torch.randn(2, 3, 64, 64, device=device)
    loss = model(small_x).square().mean()  # only for gradient smoke test
    loss.backward()
    for name, layer in [("backbone", model.stem[0]),
                        ("FPN", model.lat2), ("head", model.head[-1])]:
        grad = layer.weight.grad
        assert grad is not None and torch.isfinite(grad).all(), f"No gradient: {name}"
    print(f"Backward gradients: PASS (dummy loss={loss.item():.6f})")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
