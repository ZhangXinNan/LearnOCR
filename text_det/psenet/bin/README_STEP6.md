# LearnOCR · Step 6 — Progressive Scale Expansion (PSE)

Goal: implement **PSE from scratch in Python**, understand connected components and BFS growth, then apply the result to your trained Step 5 MiniPSENet.

This is a teaching implementation, **not** the optimized official PSENet C++/Cython implementation. The full model is also a deliberately simplified ResNet18 PSENet with **three** kernels, in the order K1 (smallest) -> K2 -> K3 (complete text).

## New files

```text
LearnOCR/
  text_detection/psenet/
    pse.py            # NumPy + OpenCV, 4-connected components and BFS expansion
    test_pse.py       # tests without dataset or model
    demo_pse.py       # synthetic visualization: touching texts with separate cores
    predict.py        # Step 5 trained checkpoint + PSE + original image output
  README_STEP6.md
```

Keep Step 2–5 files unchanged, including `model.py` and `dataset.py`. Extract the ZIP at the **LearnOCR repository root**. Run all commands from that root.

Requires existing NumPy, OpenCV and PyTorch installations. `predict.py` additionally uses `torch` and `torchvision` for Step 3 model inference; it repeats the **same mean/std** as `dataset.py` but does not import the training dataset or require pyclipper. `test_pse.py` and `demo_pse.py` do **not** require PyTorch, pyclipper or ICDAR images.

## A. Verify PSE before inference (no checkpoint needed)

```bash
python text_detection/psenet/test_pse.py
python text_detection/psenet/demo_pse.py
```

The test covers: two touching text regions but separate small seeds; nested-mask correction; tiny false positives; area/score filtering; and empty predictions.

Open `outputs/psenet_step6/synthetic_demo.png`. There are six panels: **K1, K2, K3** (binary) followed by **stage 1, stage 2, stage 3** (different colors = different instances). Importantly, **K2 and K3 already touch**, but with two separate K1 seeds PSE recovers two instances.

## B. Apply your trained Step 5 model

After the Step 5 training/overfitting experiment produces a checkpoint, run:

```bash
python text_detection/psenet/predict.py \
  --checkpoint outputs/psenet_step5/train/best.pt \
  --image data/icdar2015/test_images/img_1.jpg


python predict.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --image ../data/icdar2015/test_images/img_1.jpg
```

To start with a known image, use the overfit-one-image checkpoint and the same image:

```bash
python text_detection/psenet/predict.py \
  --checkpoint outputs/psenet_step5/overfit/last.pt \
  --image data/icdar2015/train_images/img_1.jpg


python predict.py \
  --checkpoint outputs/psenet_step5/train_960/best.pt \
  --image ../data/icdar2015/train_images/img_1.jpg

python predict2.py \
  --checkpoint outputs/psenet_step5/train_960/best.pt \
  --image ../data/icdar2015/test_images/img_1.jpg \
  --size 1280 \
  --out outputs/psenet_step6/img_1_boxes.jpg \
  --debug-dir outputs/psenet_step6/img_1_debug
```

The script reads `model_config` and `train_config.size` from the checkpoint (Step 5 format), uses the **same letterbox / ImageNet RGB normalization** as Step 5, runs sigmoid, obtains nested binary kernels, and applies PSE. It converts predicted points from the letterbox coordinate system to the **original image**. It saves:

- `outputs/psenet_step6/prediction.jpg` — original image overlaid with instance contours and min-area quadrilateral boxes
- `outputs/psenet_step6/prediction_debug.png` — letterbox image, K1/K2/K3 masks, three colored expansion stages, and filtered final instances (model resolution)
- `outputs/psenet_step6/prediction.json` — detections with confidence and quadrilateral points, in original image coordinates

For first experiments, use the same `--size` as training (automatically used from checkpoint); changing it may change detection quality. For bad/empty predictions check sigmoid heat maps in Step 5 previews first. Try `--min-kernel-area 3`, `--min-area 15`, or `--threshold 0.4` only as troubleshooting; these are tunable demonstration thresholds, not calibrated universal optimums.

**Security:** Load `.pt`/`.pth` only from your own trusted experiments. The script uses `torch.load(..., weights_only=True)`, but still don't trust unknown checkpoints.

## C. Core algorithm (read `pse.py` in this order)

1. `binarize_kernels` — compare each probability to 0.5; enforce nested masks by `K1 &= K2` and `K2 &= K3` (in reverse order K2 then K1).
2. `pse_expand` — use `cv2.connectedComponentsWithStats` on smallest K1 with **4-connectivity**. Each connected seed is a distinct text instance. Ignore tiny seeds.
3. For each larger scale, add **all currently labeled pixels** to one FIFO queue. Pop a pixel, inspect its four neighbors; claim allowed and as-yet-unlabeled neighbors for that same instance ID; push them to the queue. First arrival owns a pixel. A label cannot be stolen.
4. `extract_instances` — filter small/low-confidence regions, find their contours, compute the minimum-area rotated rectangles for ICDAR-style quadrilateral output. Save contours too, as a more natural representation for irregular text.

Pseudocode:

```text
labels = ConnectedComponents(K1)  # each seed gets an integer ID
for scale in [K2, K3]:
    queue = FIFO(all currently labeled pixels)
    while queue:
        p = queue.pop_front()
        for q in 4_neighbors(p):
            if scale[q] and labels[q] == 0:
                labels[q] = labels[p]
                queue.push_back(q)
return labels
```

**Crucial distinction:** Two text instances can occupy a *single connected component* at K3. PSE preserves separate identities from K1 through a competition for pixels. Simply doing `connectedComponents(K3)` would merge the two.

## D. Limits you should understand

- **Smallest kernels must be separate**: if K1 connects two texts, PSE **cannot** split them; if K1 misses a text altogether, it cannot recover it.
- **BFS boundary depends on seeds, thresholds, and traversal order** when larger masks connect different texts. This is a heuristic, not a learned pixel-by-pixel instance classifier.
- **Predictions are not automatically nested**; labels in Step 2 are nested, but the three output channels are independent sigmoid predictions. We explicitly intersect the thresholded masks before growing.
- **4-neighbor connectivity** is used consistently for seed labeling and expansion, a simple deterministic teaching choice. A diagonal connection alone is not treated as connected. Different production implementations may use different connectivity and filtering details.
- **Minimum area** thresholds are in input/model pixels; at small image size, small text may vanish. `min_kernel_area` filters seeds before expansion; `min_area` filters final instances afterward.
- **Confidence** here is the mean probability in K3 over the grown instance. It is not a calibrated probability that the entire word was detected correctly.
- **Contours vs boxes**: a min-area rotated rectangle is practical for ICDAR 2015 (quadrilateral output), but approximates arbitrarily shaped text. For curved text, use contours/polygons instead.
- **Performance**: pure Python BFS is intentionally easy to read, but slower than optimized C++/Cython implementations. Do not use it for benchmark FPS claims.
- **No text recognition**: output is text regions, not actual string content.
- **Pixel-Dice** from Step 5 is not text-instance F1. Instance-level Precision/Recall/F1 evaluation comes in Step 7.

## E. Learning experiments

1. Open synthetic demo and inspect all three colored stages: why does the bridge not merge IDs?
2. Change `masks[0]` so two seed cores touch, re-run: how many instances remain?
3. Set `--threshold 0.6` vs `0.4`, observe how seeds and final boundaries change.
4. Set `--min-kernel-area 50`: inspect how tiny seeds disappear.
5. Compare the predicted probability masks and final segmentation against Step 2 ground-truth masks on `img_1.jpg`.

## References

- Wang et al., *Shape Robust Text Detection with Progressive Scale Expansion Network*, CVPR 2019.
  https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html
- Official PyTorch repository: https://github.com/whai362/PSENet
