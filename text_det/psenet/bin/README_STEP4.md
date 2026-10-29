# LearnOCR / PSENet Step 4 — Loss

This step adds three files; previous Step 2/3 files remain unchanged:

- `text_detection/psenet/loss.py`: square-denominator Dice, OHEM, complete-text + kernel losses.
- `text_detection/psenet/test_loss.py`: independent tests (no ICDAR images needed).
- `text_detection/psenet/demo_loss.py`: a single real ICDAR image forward/backward verification.

## Run

From **LearnOCR repository root**:

```bash
python text_detection/psenet/test_loss.py
python text_detection/psenet/demo_loss.py --name img_1.jpg --size 256
```

The first command requires `torch` and `torchvision` (from Step 3).
The second also requires `numpy`, `opencv-python` and `pyclipper` (from Step 2), and the existing `data/icdar2015/train_images` + `train_gt`.

For devices: `--device auto|cpu|cuda|mps`.

## Shape contract

- `logits` (Step 3) = `[B,3,H,W]`, order `[K1,K2,K3]`, **raw logits**.
- `kernels` (Step 2) = `[B,3,H,W]` with 0/1 values, same order.
- `training_mask` (Step 2) = `[B,H,W]`, 1 means valid, 0 means ignore.

## Paper-style objective

Let `S_i=sigmoid(logits_i)` and `G_i=groundtruth_i`.

Square Dice: `D(P,G)=(2 sum(PG)+eps)/(sum(P²)+sum(G²)+eps)`.

- `loss_text = 1-D(S_3*M, G_3*M)` with OHEM-select mask `M`.
- `loss_kernel = mean_i(1-D(S_i*W, G_i*W))` for `i=1,2`, with `W=(S_3>=0.5) & training_mask`.
- `loss = .7*loss_text + .3*loss_kernel`.

OHEM chooses all valid positives plus up to 3 hard negative pixels for each positive; for zero positives this implementation falls back to all valid negatives. The selection is non-differentiable; the selected probabilities remain differentiable in the Dice computation.

### Important notes

1. The paper defines `W` from **predicted** complete text probabilities; this is the default `kernel_mask_mode="predicted"`. If early in training no pixels pass threshold 0.5, kernel loss can be zero because it has no selected pixels. A `kernel_mask_mode="gt"` option is provided *for diagnostic/warmup experiments only*; it is not the paper's exact loss.
2. The `training_mask` is **not** the same as the OHEM mask. Ignore regions (`###`) must be excluded from both text and kernel losses.
3. The test script checks perfect/wrong Dice, ignored-pixel invariance, OHEM hard negatives, fully ignored case, and gradients through ResNet+FPN+head.
4. `demo_loss.py` uses a square resize purely to keep the smoke test small; the full Step 5 dataset pipeline should preserve aspect ratio and keep image/polygon transforms consistent.
5. A finite loss and backward gradients only prove the training *mechanics* are connected; they do not prove accuracy or convergence. Step 5 will add optimization, validation and checkpointing.
6. Like the paper, this loss operates on sigmoid probabilities and adds a small epsilon. Images with no valid foreground are an edge case where Dice alone may provide very small learning signals for false positives; Step 5 will plan crop sampling accordingly.

Source: *Shape Robust Text Detection with Progressive Scale Expansion Network*, CVPR 2019 (Eq. 5-7): https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html
