"""Run the Step 5 MiniPSENet checkpoint + Step 6 Python PSE on one image.

From LearnOCR root:
  python text_detection/psenet/predict.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --image data/icdar2015/test_images/img_1.jpg
"""
import argparse
import json
import os.path
from pathlib import Path

import cv2
import numpy as np
import torch

from utils import choose_device
from preprocess import (bgr_to_tensor, letterbox as resize_letterbox,
                        to_original)

from model import MiniPSENet
from pse import (binarize_kernels, colorize_labels,
                 extract_instances, pse_expand)
from pse import instance_color


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


def get_args():
    p = argparse.ArgumentParser(description='MiniPSENet + Python PSE inference')
    p.add_argument('--checkpoint', #required=True,
                   default="outputs/psenet_step5/train_960/best.pt",
                   help='Step 5 best.pt / last.pt (use only trusted checkpoints)')
    p.add_argument('--image', #required=True,
                   default="../data/icdar2015/test_images/img_1.jpg")
    p.add_argument('--size', type=int, default=1280,
                   help='default: saved Step 5 train_config.size; fallback 512')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='same sigmoid threshold for all K masks')
    p.add_argument('--min-kernel-area', type=int, default=5)
    p.add_argument('--min-area', type=int, default=30,
                   help='minimum final connected area in letterbox pixels')
    p.add_argument('--min-score', type=float, default=0.5)
    p.add_argument('--out_dir', default='outputs/psenet_step6')
    return p.parse_args()


def main(args):
    # weights_only=True avoids unrestricted pickle loading for this checkpoint
    # structure; never use unknown model files just because they have a .pt suffix.
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

    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = annotate(img, detections, transform)
    save_path = out_dir / f"{Path(args.image).stem}.jpg"
    if not cv2.imwrite(save_path, overlay):
        raise RuntimeError(f'Could not write {save_path}')

    # For diagnostics, colorized instance IDs in 512x512 input coordinates.
    accepted = np.zeros(labels.shape, dtype=np.int32)
    for det in detections:
        accepted[labels == det['id']] = det['id']
    colors = colorize_labels(accepted)
    debug = [canvas]
    for k in range(masks.shape[0]):
        debug.append(cv2.cvtColor((masks[k].astype(np.uint8) * 255),
                                  cv2.COLOR_GRAY2BGR))
    # Include stage-by-stage labels as well as final filtered instances.
    debug += [colorize_labels(stage) for stage in steps]
    debug.append(colors)
    # Horizontal gallery: input | masks | expansion stages | filtered labels.
    debug_image = np.concatenate(debug, axis=1)
    debug_path = out_dir / f"{Path(args.image).stem}_debug.png"
    cv2.imwrite(str(debug_path), debug_image)

    records = []
    for det in detections:
        mapped_box = to_original(det['box'], transform)
        records.append({
            'id': det['id'], 'score': round(det['score'], 4),
            'area_at_model_resolution': det['area'],
            'box_xy': np.round(mapped_box, 1).tolist(),
        })
    json_path = out_dir / f"{Path(args.image).stem}.json"
    json_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Device: {device} | checkpoint input size: {size}')
    print(f'Detected {len(detections)} text instances')
    print(f'Overlay: {out_dir}\nDebug: {debug_path}\nQuadrilaterals: {json_path}')


if __name__ == '__main__':
    main(get_args())

