"""Run the Step 5 MiniPSENet checkpoint + Step 6 Python PSE on one image.

From LearnOCR root:
  python text_detection/psenet/predict.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --image data/icdar2015/test_images/img_1.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from preprocess import (bgr_to_tensor, letterbox as resize_letterbox,
                        to_original)

from model import MiniPSENet
from pse import (binarize_kernels, colorize_labels,
                 extract_instances, pse_expand)
from pse import instance_color
from utils import choose_device


def image_to_tensor(canvas):
    """Keep the original inference API ([1, 3, H, W])."""
    return bgr_to_tensor(canvas, batched=True)


def annotate(bgr, instances, transform):
    """Show actual instance contours AND quadrilateral boxes on original image."""
    vis = bgr.copy()
    overlay = vis.copy()
    for det in instances:
        color = tuple(int(c) for c in instance_color(det['id']))
        contour = np.rint(to_original(det['contour'], transform)).astype(np.int32)
        cv2.fillPoly(overlay, [contour], color)
    vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
    for det in instances:
        color = tuple(int(c) for c in instance_color(det['id']))
        polygon = np.rint(to_original(det['box'], transform)).astype(np.int32)
        cv2.polylines(vis, [polygon], True, color, 2)
        x, y = polygon.min(axis=0)
        cv2.putText(vis, f"#{det['id']} {det['score']:.2f}",
                    (int(x), max(20, int(y) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


def probability_heatmap(probability):
    """Visualize one probability map [H,W] as a JET heatmap."""
    img = np.clip(np.rint(probability * 255.0), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)


def binary_mask_image(mask):
    """Visualize one binary mask [H,W] as white foreground on black bg."""
    gray = (mask.astype(np.uint8) * 255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def save_debug_artifacts(debug_dir, original_bgr, canvas, valid, probabilities,
                         masks, steps, instances, labels, transform):
    """Save the requested intermediate PSENet visualization results."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 0. Original and letterbox input.
    cv2.imwrite(str(debug_dir / '00_original.jpg'), original_bgr)
    cv2.imwrite(str(debug_dir / '01_letterbox_input.jpg'), canvas)
    cv2.imwrite(str(debug_dir / '01_letterbox_valid_mask.png'),
                binary_mask_image(valid))

    # 1. Three raw probability heatmaps.
    for i in range(probabilities.shape[0]):
        heat = probability_heatmap(probabilities[i])
        blend = cv2.addWeighted(canvas, 0.45, heat, 0.55, 0)
        cv2.imwrite(str(debug_dir / f'10_prob_k{i + 1}_heatmap.png'), heat)
        cv2.imwrite(str(debug_dir / f'11_prob_k{i + 1}_overlay.png'), blend)

    # 2. Three binary masks after thresholding.
    for i in range(masks.shape[0]):
        cv2.imwrite(str(debug_dir / f'20_mask_k{i + 1}.png'),
                    binary_mask_image(masks[i]))

    # 3. Three PSE stages (instance ID label maps).
    # steps[0] = seeds from K1, steps[1] = after growth in K2, ...
    for i, stage in enumerate(steps, start=1):
        colored = colorize_labels(stage)
        cv2.imwrite(str(debug_dir / f'30_pse_stage_{i}.png'), colored)

    # 4. Final accepted instances on model-resolution label map.
    accepted = np.zeros(labels.shape, dtype=np.int32)
    for det in instances:
        accepted[labels == det['id']] = det['id']
    cv2.imwrite(str(debug_dir / '31_pse_final_accepted.png'),
                colorize_labels(accepted))

    # 5. Final detection result on the original image.
    overlay = annotate(original_bgr, instances, transform)
    cv2.imwrite(str(debug_dir / '40_final_boxes.jpg'), overlay)

    # 6. One compact horizontal gallery for quick browsing.
    gallery = [canvas]
    gallery.extend(probability_heatmap(probabilities[i]) for i in range(probabilities.shape[0]))
    gallery.extend(binary_mask_image(masks[i]) for i in range(masks.shape[0]))
    gallery.extend(colorize_labels(stage) for stage in steps)
    gallery.append(colorize_labels(accepted))
    gallery.append(cv2.resize(overlay, (canvas.shape[1], canvas.shape[0])))
    gallery_image = np.concatenate(gallery, axis=1)
    cv2.imwrite(str(debug_dir / '99_gallery.png'), gallery_image)


def main():
    p = argparse.ArgumentParser(description='MiniPSENet + Python PSE inference')
    p.add_argument('--checkpoint', required=True,
                   help='Step 5 best.pt / last.pt (use only trusted checkpoints)')
    p.add_argument('--image', required=True)
    p.add_argument('--size', type=int, default=None,
                   help='default: saved Step 5 train_config.size; fallback 512')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='same sigmoid threshold for all K masks')
    p.add_argument('--min-kernel-area', type=int, default=5)
    p.add_argument('--min-area', type=int, default=30,
                   help='minimum final connected area in letterbox pixels')
    p.add_argument('--min-score', type=float, default=0.5)
    p.add_argument('--out', default='outputs/psenet_step6/prediction.jpg')
    p.add_argument('--debug-dir', default=None,
                   help='Directory to save heatmaps, masks, PSE stages and final boxes. '
                        'Default: <out stem>_debug/')
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = ckpt.get('model_config', {})
    train_config = ckpt.get('train_config', {})
    size = args.size or int(train_config.get('size', 512))
    device = choose_device(args.device)
    model = MiniPSENet(
        num_kernels=int(config.get('num_kernels', 3)),
        fpn_channels=int(config.get('fpn_channels', 64)),
        pretrained=False,
    )
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.to(device).eval()

    img = cv2.imread(str(args.image),
                     cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(args.image)
    canvas, valid, transform = resize_letterbox(img, size)
    x = image_to_tensor(canvas).to(device)

    with torch.inference_mode():
        probabilities = model(x).sigmoid()[0].float().cpu().numpy()
    probabilities[:, ~valid] = 0.0  # don't detect in letterbox padding
    masks = binarize_kernels(probabilities, args.threshold)
    labels, steps = pse_expand(
        masks, min_kernel_area=args.min_kernel_area, return_steps=True
    )
    detections = extract_instances(
        labels, probabilities[-1], min_area=args.min_area, min_score=args.min_score
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    overlay = annotate(img, detections, transform)
    if not cv2.imwrite(str(out), overlay):
        raise RuntimeError(f'Could not write {out}')

    debug_dir = Path(args.debug_dir) if args.debug_dir else out.with_name(out.stem + '_debug')
    save_debug_artifacts(debug_dir, img, canvas, valid, probabilities,
                         masks, steps, detections, labels, transform)

    records = []
    for det in detections:
        mapped_box = to_original(det['box'], transform)
        records.append({
            'id': det['id'], 'score': round(det['score'], 4),
            'area_at_model_resolution': det['area'],
            'box_xy': np.round(mapped_box, 1).tolist(),
        })
    json_path = out.with_suffix('.json')
    json_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Device: {device} | checkpoint input size: {size}')
    print(f'Detected {len(detections)} text instances')
    print(f'Final overlay: {out}')
    print(f'Debug directory: {debug_dir}')
    print(f'Quadrilaterals: {json_path}')


if __name__ == '__main__':
    main()
