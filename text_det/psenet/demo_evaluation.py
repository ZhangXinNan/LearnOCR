"""Visualize Step 7 TP / FP / FN / ignore using synthetic, non-model detections."""
from pathlib import Path

import cv2
import numpy as np

from eval_metrics import evaluate_image
from evaluate import render


def rectangle(x, y, w, h):
    return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]


def main():
    img = np.full((280, 700, 3), 245, dtype=np.uint8)
    for word, x, y in [('MATCHED', 50, 93), ('MISSED', 290, 93),
                       ('IGNORE', 520, 93)]:
        cv2.putText(img, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, (40, 40, 40), 2, cv2.LINE_AA)
    gts = [rectangle(45, 48, 170, 54), rectangle(280, 48, 170, 54),
           rectangle(510, 48, 170, 54)]
    ignored = [False, False, True]
    predictions = [
        {'box_xy': rectangle(47, 50, 166, 50), 'score': 0.96},  # TP
        {'box_xy': rectangle(285, 160, 140, 52), 'score': 0.72}, # FP
        {'box_xy': rectangle(514, 52, 162, 48), 'score': 0.91},  # ignore
    ]
    result = evaluate_image(gts, ignored, predictions)
    output = render(img, gts, ignored, predictions, result)
    cv2.putText(output, 'Green=TP  Red=FP  Orange=FN  Grey=ignored  Cyan=matched GT',
                (25, 255), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (40, 40, 40), 2, cv2.LINE_AA)
    path = Path('outputs/psenet_step7/synthetic_demo.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), output)
    print('Synthetic metrics:', {k: result[k] for k in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')})
    print('Saved:', path)


if __name__ == '__main__':
    main()
