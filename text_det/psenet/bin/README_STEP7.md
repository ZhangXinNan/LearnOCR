# LearnOCR / PSENet — Step 7: Evaluation & Visualization

Goal: **Evaluate word-level text detection, not Pixel Dice**, and inspect false positives and false negatives.

## 1. Files added

```
text_detection/psenet/
  eval_metrics.py       # polygon IoU, ### filtering, one-to-one matching, P/R/F1
  evaluate.py           # checkpoint -> full dataset -> predictions/metrics/images
  test_evaluation.py    # 8 tests; no ICDAR dataset needed
  demo_evaluation.py    # synthetic visual explanation of TP / FP / FN / ignore
  plot_evaluation.py    # summary plots from evaluation output
```

Requires Step 3 `model.py`, Step 6 `pse.py` and `predict.py`, and a checkpoint produced by Step 5. Run commands **from the LearnOCR repository root**. Reuse your existing PyTorch/NumPy/OpenCV/Matplotlib environment.

## 2. Understand the evaluation protocol

- **GT**: ICDAR 2015 word-level quadrilaterals (`gt_img_1.txt`).
- **Prediction**: one quadrilateral per PSE instance in **original image coordinates**.
- **IoU** = intersection area / union area of predicted and GT quadrilaterals.
- **TP**: one matched prediction and one *care* GT with `IoU > 0.5` (strictly greater).
- **FP**: non-ignored prediction not matched to a care GT (e.g., false text, duplicate boxes).
- **FN**: care GT with no matching prediction.
- **`###`** = do-not-care GT. A prediction is excluded from scoring if **intersection / prediction area > 0.5** with a `###` region. This differs from simply dropping all predictions that touch a `###` box.
- **Precision** = TP/(TP+FP); **Recall** = TP/(TP+FN); **F1** = 2PR/(P+R).
- Sum TP/FP/FN **across images before computing overall scores** (micro-average); do not average per-image F1 values to get the dataset F1.

The ICDAR 2015 competition uses an IoU 0.5 protocol. **Our local evaluator is educational, not byte-for-byte official**: it uses greedy largest-IoU one-to-one matching, while the RRC script may apply another tie/order policy. `###` handling is modeled after the common RRC script. Use the **official RRC evaluation tool** to obtain benchmark-comparable numbers. Reproducing official results also requires matching all official formatting/validation rules.

References:
- https://rrc.cvc.uab.es/?ch=4&com=faq
- https://rrc.cvc.uab.es/?ch=4&com=tasks

## 3. Verify evaluation logic first

```bash
python text_detection/psenet/test_evaluation.py
python text_detection/psenet/demo_evaluation.py
```

Expected synthetic demo output: TP=1, FP=1, FN=1, P=R=F1=0.5. Output picture:
`outputs/psenet_step7/synthetic_demo.png`.

Legend: green TP prediction, red FP prediction, orange missed GT, gray ignored, cyan matched GT.

## 4. Quick validation smoke test

You must have the trained Step 5 files:

```
outputs/psenet_step5/train/best.pt
outputs/psenet_step5/train/split.json
```

Start with 5 held-out validation images:

```bash
python text_detection/psenet/evaluate.py \
  --checkpoint outputs/psenet_step5/train/best.pt \
  --split val --max-images 5 --visualize 5
  
python evaluate.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --split val \
    --max-images 5 \
    --visualize 5
```

The first run may be slow: Python BFS is a learning implementation.

## 5. Evaluate all held-out validation images

```bash
python text_detection/psenet/evaluate.py \
  --checkpoint outputs/psenet_step5/train/best.pt \
  --split val --visualize 12
  
python evaluate.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --split val \
    --visualize 12


python text_detection/psenet/plot_evaluation.py \
  --dir outputs/psenet_step7/validation
  
python plot_evaluation.py \
  --dir outputs/psenet_step7/validation
```

The `split.json` from **Step 5 train mode** determines which images belong to the held-out validation subset (typically 100 of 1,000). Do not re-split with a different random seed. If evaluating an overfit checkpoint, the default train split JSON won't correspond to it; do the single-image visual check with Step 6 `predict.py` instead.

## 6. Optional final evaluation on ICDAR2015 official test images

Use this **after** validating and choosing thresholds on validation data, not for repeated tuning:

```bash
python evaluate.py \
  --checkpoint outputs/psenet_step5/train/best.pt \
  --split test \
  --out outputs/psenet_step7/test
```

Requires `data/icdar2015/test_images` and `data/icdar2015/test_gt`. If you do not have test ground truth, you can still run individual inference with Step 6 `predict.py`, but cannot compute local F1 on the test split. `outputs/psenet_step7/test/rrc_results.zip` contains RRC-style `res_img_N.txt` predictions. Submit/check with the official RRC tool for an official benchmark score; **this script does not run the official scoring package**.

## 7. Outputs

```
outputs/psenet_step7/validation/
  summary.json                # dataset-wide micro P/R/F1 + eval parameters
  per_image.csv               # per-image P/R/F1 and TP/FP/FN
  details.json                # boxes, matches, error indices
  visualizations/             # lowest per-image F1 cases (care GT only)
  rrc_results/                # res_img_N.txt per image
  rrc_results.zip             # same predictions, single ZIP
  metrics.png                 # created by plot_evaluation.py
  f1_distribution.png         # created by plot_evaluation.py
```

Note: `--max-images 5` evaluates only five images and **must not be reported as the full validation score**. For the official test, use `--max-images 0` (the default).

## 8. Learning experiments

1. Compare `--threshold 0.4` vs `0.5` vs `0.6` on validation. Explain changes in FP and FN, not just F1.
2. Change `--min-kernel-area` and `--min-area`: small text vs noisy seeds.
3. Inspect the 12 lowest-F1 visualizations. Categorize **small/blurred text**, **adjacent words fused**, **text split**, **background false positive**, **localization IoU below 0.5**.
4. Compare training Pixel Dice and instance F1. Explain why a high pixel overlap does not guarantee distinct word instances.
5. Record final parameters and one sample visualization in `text_detection/psenet/README.md`.

### Why PSENet may still underperform

This is a simplified teaching model (ResNet18, 3 kernels, narrow FPN, short training), not the CVPR2019 reproduction. Low recall and merged/split text are valuable diagnostic signals; do not claim the original paper's results.
