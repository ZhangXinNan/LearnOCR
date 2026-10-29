# LearnOCR · PSENet Step 5 — Training Pipeline

Goal: connect Step 2 (`kernel.py`), Step 3 (`model.py`), and Step 4 (`loss.py`) into a minimal trainable PSENet. **This is a teaching implementation, not an exact reproduction of CVPR 2019.**

## Files (new in Step 5)

- `text_detection/psenet/dataset.py`: reads ICDAR 2015 annotations; letterboxes images without changing aspect ratio; transforms polygons; generates K1/K2/K3 labels *after* transform; ignores padding and `###` regions; optional photometric augmentation.
- `text_detection/psenet/train.py`: overfit-one-image experiment, 90/10 held-out validation for full training, Dice/OHEM loss, AdamW, pixel-Dice metric, CSV logs, prediction visualizations, checkpoints and optional resume.
- `text_detection/psenet/test_training.py`: generates a synthetic labelled image in a temporary directory; tests mask alignment and one gradient/optimizer update.
- `text_detection/psenet/plot_history.py`: plots loss and pixel-Dice curves from the CSV file.

Keep existing Step 2–4 files untouched. Run all commands **from LearnOCR repository root** (so `data/icdar2015` resolves).

## Requirements

Python 3.10+, PyTorch + torchvision, OpenCV, NumPy, pyclipper. `--pretrained` uses torchvision ImageNet weights and may need a one-time network download. Otherwise no download is required beyond ICDAR 2015.

```
python text_detection/psenet/test_training.py
```

## Stage A — overfit just one picture

```
python text_detection/psenet/train.py \
  --mode overfit --name img_1.jpg --size 320 \
  --overfit-steps 250 --lr 0.0003 --save-every 50
  
python train.py \
  --mode overfit \
  --name img_1.jpg \
  --size 320 \
  --overfit-steps 250 \
  --lr 0.0003 \
  --save-every 50
```

First inspect `outputs/psenet_step5/overfit/preview_step_0000.jpg`, and compare with `preview_step_0050.jpg`, etc. Each preview displays input and GT K1/K2/K3 (top row) and predicted kernel probabilities (bottom row). `history.csv` contains loss, text loss, kernel loss and **thresholded pixel Dice (NOT detection F1)**.

Plot the training curves after the experiment:

```
python text_detection/psenet/plot_history.py \
  --history outputs/psenet_step5/overfit/history.csv
```

The overfit experiment uses **the same one image repeatedly**: its metric is a diagnostic, not generalization performance. Progress is not guaranteed at exactly 250 steps; try increasing steps or reducing LR if unstable. A randomly initialized model can learn one picture even though it does not yet generalize.

If the chosen picture becomes all-ignored at small size (no valid K3 text pixels), choose another `--name` or larger `--size`.

## Stage B — train on ICDAR 2015

```bash
python text_detection/psenet/train.py \
  --mode train --pretrained --size 512 --batch-size 2 \
  --epochs 20 --lr 0.0001


python train.py \
    --mode train \
    --pretrained \
    --size 512 \
    --batch-size 2 \
    --epochs 20 \
    --lr 0.0001
```

- First split the original **1,000 ICDAR 2015 training images** reproducibly into 900 training + 100 validation images, with `--val-fraction 0.1 --seed 42`. Official test images are not used during training.
- Validation has **no augmentation** and is not part of optimization.
- Photometric augmentation is intentionally mild; no horizontal flip / random crop / 90-degree rotations yet. Data diversity and accuracy can improve later with careful geometric augmentations.
- `--pretrained` is recommended for full training because 900 images is small for training an entire ResNet18 from random initialization. ImageNet normalization matches these weights.
- To plot full training curves, use `--history outputs/psenet_step5/train/history.csv` with `plot_history.py`.
- More epochs or augmentations may be needed; this configuration is for learning the full training loop, not reproducing published detection metrics.

Resources: for slower devices, try `--size 320 --batch-size 1`, accepting reduced text detail. On Mac M-series, `--device mps` (or auto) selects Metal; on NVIDIA, `--device cuda` (or auto) selects CUDA. CPU also works but is slower. No AMP is used initially so loss/gradient behavior is easier to debug.

## Files generated

```
outputs/psenet_step5/overfit/
  split.json
  history.csv
  preview_step_0000.jpg
  preview_step_0050.jpg ...
  best.pt
  last.pt

outputs/psenet_step5/train/
  split.json                # exact train/validation filenames
  history.csv
  val_preview_epoch_001.jpg ...
  best.pt                  # lowest validation loss so far
  last.pt                  # most recent completed epoch
```

Checkpoint contents: `model_state`, `optimizer_state`, `step`, `epoch`, `model_config`, `train_config`, `best_value`.
To continue within the same output folder (only use checkpoints you trust):

```
python text_detection/psenet/train.py --mode train \
  --pretrained --size 512 --batch-size 2 --epochs 30 --lr 0.0001 \
  --resume outputs/psenet_step5/train/last.pt
```

In this example `--epochs 30` means **train up to epoch 30**, not train 30 additional epochs. Keep the same `--seed`, `--size`, data path and hyperparameters when resuming. The script saves the split with a fixed seed but does not yet assert that a resumed run's dataset/hyperparameters match the original. For exact mid-epoch reproducibility, RNG state and DataLoader worker state would also need saving; this simple version resumes at epoch boundaries.

## How to reason about results

1. `loss` decreasing indicates progress on the chosen optimization objective; it does not alone prove correct detection.
2. `val_pixel_Dice` measures binary pixel overlap of the *complete text* channel K3, thresholded at 0.5 and masked to valid areas. It **does not** measure instance detection Precision/Recall/F1. Those require PSE postprocessing (Step 6) and matching text polygons to ground truth (Step 7).
3. K1/K2 output probabilities do not have to be perfectly nested even though targets are nested.
4. With the Step 4 paper-style `kernel_mask_mode="predicted"`, early kernel loss can be zero if K3 probabilities are all below 0.5. You may run `--kernel-mask-mode gt` as a **clearly marked diagnostic** but do not compare it as an identical paper implementation.
5. PSENet's ground-truth shrink procedure is repeated after resizing; polygons too small to form K1 are ignored to avoid inconsistent labels.
6. The masks are binary and pixels marked `training_mask=0` (including padding) do **not** affect loss.

## Suggested learning experiments

- Compare `preview_step_0000.jpg` and `preview_step_0200.jpg`.
- Try 128/320/512 size (all sizes must be multiples of 32, minimum 64); inspect tiny texts lost from downsizing.
- Try `--kernel-mask-mode predicted` vs `gt` with different output folders, and compare K1/K2 early training behavior.
- After overfitting works, use the full train/validation split; inspect whether training loss falls while validation loss stays high (overfitting).

Next Step 6: PSE postprocessing — connected components from K1, progressively expand to K2/K3, extract text polygons.
