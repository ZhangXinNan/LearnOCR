# LearnOCR / PSENet Step 3 — Minimal Network

This adds **only two files** to the existing Step 2 project:

- `text_detection/psenet/model.py` — PyTorch `MiniPSENet` model.
- `text_detection/psenet/test_model.py` — standalone smoke test.

No data files or Step 2 code are overwritten.

## Run

Install compatible `torch` and `torchvision` from <https://pytorch.org/get-started/locally/>.

From the **LearnOCR repository root**:

```bash
python text_detection/psenet/test_model.py
python text_detection/psenet/test_model.py --height 640 --width 640
```

Smoke test performs a forward pass at the requested size (default 256x384) and a small 64x64 backward pass, and asserts that gradient reaches the backbone, FPN and head.

### Shapes for input [1, 3, 640, 640]

| Component | Shape |
| --- | --- |
| C2 | [1, 64, 160, 160] |
| C3 | [1, 128, 80, 80] |
| C4 | [1, 256, 40, 40] |
| C5 | [1, 512, 20, 20] |
| P2..P5 (different spatial sizes) | 64 channels each |
| fused | [1, 256, 160, 160] |
| small_logits | [1, 3, 160, 160] |
| logits | [1, 3, 640, 640] |

Output channel ordering follows Step 2 `RATIOS=(0.50, 0.75, 1.00)`:

- index 0: K1, smallest kernel
- index 1: K2, intermediate kernel
- index 2: K3, full text region

### Important design choices

- `model(x)` returns **raw logits**, not probabilities. For visualization only, use `torch.sigmoid(logits)`. Later, the Dice/BCE loss code must handle logits appropriately.
- Model predictions are **not automatically nested**. The training labels are nested, but raw predictions need not be. Post-processing will enforce containment before progressive scale expansion.
- `pretrained=False` avoids a download so smoke testing works offline. For practical training, consider `MiniPSENet(pretrained=True)`, which downloads ImageNet weights the first time (requires Internet and increases quality/convergence). It does **not** download anything when `pretrained=False`.
- Input `H,W` are recommended multiples of 32 for predictable backbone shapes, but output is explicitly interpolated to original `H,W` (including odd sizes).
- This is an educational **simplification** of PSENet, not an exact reproduction of the original paper. It uses ResNet18, 64-channel simplified FPN and 3 output kernels instead of the heavier official configuration.
- The smoke test's `loss = logits.square().mean()` is **not** the real PSENet loss. The real text/kernel training losses come in Step 4.

## How Step 2 and Step 3 connect

```
Step 2 labels: [B, 3, H, W] (K1, K2, K3)
Step 3 logits: [B, 3, H, W] (K1, K2, K3)
Training mask: [B, H, W]
```

The next step will implement differentiable Dice losses and a masked training objective.
