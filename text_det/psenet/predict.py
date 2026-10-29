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

# Match the ImageNet RGB normalization used by Step 5 dataset.py.
# Define it here to avoid requiring dataset/pyclipper during inference.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

from model import MiniPSENet
from pse import (binarize_kernels, colorize_labels,
                 extract_instances, pse_expand)


def choose_device(name):
    if name != 'auto':
        if name == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if name == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable')
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resize_letterbox(bgr, size):
    """Match Step 5 dataset exactly; return canvas and reverse-map metadata."""
    h, w = bgr.shape[:2]
    if size < 64 or size % 32 != 0:
        raise ValueError('Input size must be >=64 and divisible by 32')
    scale = min(size / w, size / h)
    new_w = max(1, min(size, round(w * scale)))
    new_h = max(1, min(size, round(h * scale)))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    valid = np.zeros((size, size), dtype=bool)
    valid[top:top + new_h, left:left + new_w] = True
    return canvas, valid, (new_w / w, new_h / h, left, top, w, h)


def image_to_tensor(canvas):
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)


def to_original(points, transform):
    sx, sy, left, top, width, height = transform
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] = np.clip((pts[:, 0] - left) / sx, 0, width - 1)
    pts[:, 1] = np.clip((pts[:, 1] - top) / sy, 0, height - 1)
    return pts


def annotate(bgr, instances, transform):
    """Show actual instance contours AND quadrilateral boxes on original image."""
    vis = bgr.copy()
    overlay = vis.copy()
    for det in instances:
        color = tuple(int(c) for c in colorize_id(det['id']))
        contour = np.rint(to_original(det['contour'], transform)).astype(np.int32)
        cv2.fillPoly(overlay, [contour], color)
    vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
    for det in instances:
        color = tuple(int(c) for c in colorize_id(det['id']))
        polygon = np.rint(to_original(det['box'], transform)).astype(np.int32)
        cv2.polylines(vis, [polygon], True, color, 2)
        x, y = polygon.min(axis=0)
        cv2.putText(vis, f"#{det['id']} {det['score']:.2f}",
                    (int(x), max(20, int(y) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


def colorize_id(instance_id):
    n = int(instance_id)
    return (60 + (67 * n) % 196, 60 + (113 * n) % 196,
            60 + (157 * n) % 196)


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
    args = p.parse_args()

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
    debug_path = out.with_name(out.stem + '_debug.png')
    cv2.imwrite(str(debug_path), debug_image)

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
    print(f'Overlay: {out}\nDebug: {debug_path}\nQuadrilaterals: {json_path}')


if __name__ == '__main__':
    main()
