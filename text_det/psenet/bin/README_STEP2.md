# LearnOCR / PSENet / Step 2 — Kernel Label Generation

Goal: turn one ICDAR 2015 quadrilateral label file into three nested binary text masks.

We keep `ratios = (0.5, 0.75, 1.0)` as a simple teaching configuration. The three masks represent increasingly complete text regions. This is a simplified PSENet configuration, not a strict reproduction of the official training setup.

## Files

- `text_detection/psenet/kernel.py`: read ICDAR 2015 annotations, shrink polygons with `pyclipper`, make kernel and training masks.
- `text_detection/psenet/visualize_kernels.py`: inspect a sample image and save five panels.

## Setup

From your **LearnOCR repository root**, copy these two `.py` files into `text_detection/psenet/`, and make sure you have:

```
data/icdar2015/train_images/img_1.jpg
data/icdar2015/train_gt/gt_img_1.txt
```

Install dependencies if needed:

```bash
python -m pip install numpy opencv-python pyclipper matplotlib
```

Check polygon-shrink logic with the synthetic built-in test:

```bash
python text_detection/psenet/kernel.py
```

Visualize a real ICDAR 2015 sample:

```bash
python text_detection/psenet/visualize_kernels.py
```

Open `outputs/psenet_step2.png`.

For another image:

```bash
python text_detection/psenet/visualize_kernels.py --name img_2.jpg
```

If your dataset is stored elsewhere, add `--data /your/dataset/path`.

## What to verify

1. K1 is thin and mostly separates adjacent words.
2. K2 is bigger; K3 corresponds to original text polygons.
3. Masks are nested: `K1 ⊆ K2 ⊆ K3`.
4. In the `Training mask`, black indicates `###` regions or unshrinkable tiny instances; those pixels are excluded when loss is computed.

All masks are saved as `uint8` in memory, and use **0/1**, not 0/255. Later the model will predict three logits channels matched to these three masks.

## Core PSENet formula

For polygon area `A`, perimeter `L`, and scale ratio `r`:

`distance = A * (1 - r*r) / L`

`pyclipper` then performs an inward polygon offset by `-distance`. `r` is a scale parameter used by this formula; it is not a direct x/y scaling factor.
