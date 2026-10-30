"""Step 7: evaluate MiniPSENet + PSE on the held-out ICDAR2015 validation set.

Run from LearnOCR repository root.  See README_STEP7.md.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from eval_metrics import (counts_to_metrics, evaluate_image, read_icdar_annotations)
from model import MiniPSENet
from predict import (choose_device, image_to_tensor, resize_letterbox, to_original)
from pse import binarize_kernels, extract_instances, pse_expand


def get_images(data_root, split, split_json):
    data_root = Path(data_root)
    if split == 'test':
        image_dir, gt_dir = data_root / 'test_images', data_root / 'test_gt'
        names = sorted(p.name for p in image_dir.iterdir()
                       if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    else:
        image_dir, gt_dir = data_root / 'train_images', data_root / 'train_gt'
        if not Path(split_json).is_file():
            raise FileNotFoundError(f'Missing split: {split_json}; Step 5 train mode creates it')
        split_info = json.loads(Path(split_json).read_text(encoding='utf-8'))
        key = 'validation_images' if split == 'val' else 'train_images'
        if key not in split_info:
            raise KeyError(f'Step 5 split.json lacks {key}; use a train-mode checkpoint')
        names = split_info[key]
    if not names:
        raise ValueError(f'No {split} images found')
    for name in names:
        image_path = image_dir / name
        gt_path = gt_dir / f'gt_{Path(name).stem}.txt'
        if not image_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(f'Missing image/GT pair: {image_path} / {gt_path}')
    return [(name, image_dir / name,
             gt_dir / f'gt_{Path(name).stem}.txt') for name in names]


def load_model(checkpoint, device):
    # Only load trusted checkpoints created by your training; never random .pt files.
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model_cfg = ckpt.get('model_config', {})
    model = MiniPSENet(
        num_kernels=int(model_cfg.get('num_kernels', 3)),
        fpn_channels=int(model_cfg.get('fpn_channels', 64)),
        pretrained=False,
    )
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.to(device).eval()
    input_size = int(ckpt.get('train_config', {}).get('size', 512))
    return model, input_size


def predict_boxes(model, image, size, device, args):
    canvas, valid, transform = resize_letterbox(image, size)
    x = image_to_tensor(canvas).to(device)
    with torch.inference_mode():
        probabilities = model(x).sigmoid()[0].float().cpu().numpy()
    probabilities[:, ~valid] = 0.0
    binary = binarize_kernels(probabilities, args.threshold)
    labels = pse_expand(binary, min_kernel_area=args.min_kernel_area)
    instances = extract_instances(labels, probabilities[-1],
                                  min_area=args.min_area, min_score=args.min_score)
    return [{'box_xy': to_original(inst['box'], transform).tolist(),
             'score': inst['score']} for inst in instances]


def render(image, gt_polygons, ignored, predictions, evaluation):
    """Green=TP det; red=FP det; orange=FN GT; gray=ignored; cyan=matched GT."""
    out = image.copy()
    matched_gt = set(evaluation['matched_gt'])
    matched_det = set(evaluation['matched_det'])
    ignored_det = set(evaluation['ignored_det'])

    def line(points, color, label, thickness=2):
        pts = np.rint(np.asarray(points)).astype(np.int32)
        cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA)
        x, y = pts.min(axis=0)
        cv2.putText(out, label, (max(0, int(x)), max(16, int(y) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    for i, (polygon, ignore) in enumerate(zip(gt_polygons, ignored)):
        if ignore:
            line(polygon, (140, 140, 140), 'ignore', 1)
        elif i in matched_gt:
            line(polygon, (255, 255, 0), f'GT{i}', 1)
        else:
            line(polygon, (0, 165, 255), f'FN{i}', 2)
    for j, det in enumerate(predictions):
        color = ((128, 128, 128) if j in ignored_det else
                 (0, 190, 0) if j in matched_det else (0, 0, 255))
        status = 'IGN' if j in ignored_det else 'TP' if j in matched_det else 'FP'
        line(det['box_xy'], color, f'{status}{j}:{det["score"]:.2f}', 2)
    cv2.rectangle(out, (0, 0), (min(out.shape[1] - 1, 470), 34), (18, 18, 18), -1)
    cv2.putText(out, f'TP={evaluation["tp"]} FP={evaluation["fp"]} FN={evaluation["fn"]}  '
                f'F1={evaluation["f1"]:.3f}', (7, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_rrc_result(file_path, predictions):
    """RRC ICDAR15 quadrilateral lines: x1,y1,...,x4,y4 (no confidence)."""
    with Path(file_path).open('w', encoding='utf-8', newline='\n') as f:
        for det in predictions:
            xy = np.rint(np.asarray(det['box_xy'])).astype(np.int32).reshape(8)
            f.write(','.join(str(int(x)) for x in xy) + '\n')


def main():
    p = argparse.ArgumentParser(description='LearnOCR PSENet word-level evaluation')
    p.add_argument('--checkpoint', default='outputs/psenet_step5/train/best.pt')
    p.add_argument('--data', default='../data/icdar2015')
    p.add_argument('--split', choices=('val', 'train', 'test'), default='val')
    p.add_argument('--split-json', default='outputs/psenet_step5/train/split.json')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    p.add_argument('--size', type=int, default=None, help='default: checkpoint size')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--min-kernel-area', type=int, default=5)
    p.add_argument('--min-area', type=int, default=30)
    p.add_argument('--min-score', type=float, default=0.5)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--max-images', type=int, default=0,
                   help='0=all; positive integer for smoke testing only')
    p.add_argument('--visualize', type=int, default=12,
                   help='save this many lowest-F1 examples with at least one care GT')
    p.add_argument('--out', default='outputs/psenet_step7')
    args = p.parse_args()
    if args.max_images < 0 or args.visualize < 0:
        p.error('--max-images and --visualize must be >= 0')
    if args.min_area < 1 or args.min_kernel_area < 1:
        p.error('minimum areas must be positive')
    if not (0 < args.threshold < 1 and 0 <= args.min_score <= 1 and 0 < args.iou < 1):
        p.error('threshold and iou must be in (0,1); min-score in [0,1]')

    device = choose_device(args.device)
    model, checkpoint_size = load_model(args.checkpoint, device)
    size = args.size or checkpoint_size
    image_pairs = get_images(args.data, args.split, args.split_json)
    if args.max_images:
        image_pairs = image_pairs[:args.max_images]

    weights_dirname = Path(args.checkpoint).parent.name
    out = Path(args.out) / f"{weights_dirname}.{args.split}.{args.size}"
    rrc_dir = out / 'rrc_results'
    rrc_dir.mkdir(parents=True, exist_ok=True)
    rows, total = [], {'tp': 0, 'fp': 0, 'fn': 0}
    print(f'Device={device}; split={args.split}; images={len(image_pairs)}; size={size}', flush=True)

    for count, (name, image_file, gt_file) in enumerate(image_pairs, 1):
        image = cv2.imread(str(image_file),
                           cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(image_file)
        gt_polygons, ignored = read_icdar_annotations(gt_file)
        detections = predict_boxes(model, image, size, device, args)
        result = evaluate_image(gt_polygons, ignored, detections,
                                iou_threshold=args.iou)
        for metric in total:
            total[metric] += result[metric]
        # Save per-image results for reproducibility and offline inspection.
        row = {'name': name, **{k: result[k] for k in
               ('tp', 'fp', 'fn', 'num_care_gt', 'num_care_det',
                'precision', 'recall', 'f1')},
               'predictions': detections, 'evaluation': result}
        rows.append(row)
        write_rrc_result(rrc_dir / f'res_{Path(name).stem}.txt', detections)
        if count % 10 == 0 or count == len(image_pairs):
            print(f'  {count}/{len(image_pairs)} images', flush=True)

    metrics = counts_to_metrics(**total)
    summary = {
        'split': args.split, 'images': len(image_pairs), 'checkpoint': str(args.checkpoint),
        'input_size': size, 'threshold': args.threshold, 'min_score': args.min_score,
        'min_area': args.min_area, 'min_kernel_area': args.min_kernel_area,
        'iou_threshold': args.iou, 'ignore_overlap_threshold': 0.5,
        'matching': 'greedy max IoU (educational; not the official RRC score)',
        **total, **metrics,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out / 'per_image.csv').open('w', newline='', encoding='utf-8') as f:
        cols = ('name', 'tp', 'fp', 'fn', 'num_care_gt', 'num_care_det',
                'precision', 'recall', 'f1')
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in cols})
    (out / 'details.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')

    # Save only the most difficult images (not random easy samples).
    viz_dir = out / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    lookup = {name: (im, gt) for name, im, gt in image_pairs}
    difficult = sorted((r for r in rows if r['num_care_gt'] > 0),
                       key=lambda r: (r['f1'], -r['fn'], -r['fp'], r['name']))
    for row in difficult[:args.visualize]:
        image_file, gt_file = lookup[row['name']]
        img = cv2.imread(str(image_file),
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        gts, ignored = read_icdar_annotations(gt_file)
        vis = render(img, gts, ignored, row['predictions'], row['evaluation'])
        path = viz_dir / row['name']
        if not cv2.imwrite(str(path), vis):
            raise RuntimeError(f'Could not write visualization: {path}')

    zip_path = out / 'rrc_results.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(rrc_dir.glob('res_*.txt')):
            z.write(path, arcname=path.name)
    print(f'\nTP={total["tp"]}, FP={total["fp"]}, FN={total["fn"]}')
    print(f'Precision={metrics["precision"]:.4f}  '
          f'Recall={metrics["recall"]:.4f}  F1={metrics["f1"]:.4f}')
    if args.max_images:
        print('WARNING: partial dataset (--max-images); diagnostic metrics only')
    if args.split != 'test':
        print('RRC ZIP is for inspection; benchmark submissions require full test split.')
    print(f'Saved: {out}/summary.json, per_image.csv, details.json, '
          f'visualizations/, rrc_results.zip')


if __name__ == '__main__':
    main()
