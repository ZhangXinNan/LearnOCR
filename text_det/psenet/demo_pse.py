"""Visualize staged PSE expansion on TWO touching synthetic text regions."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from pse import colorize_labels, pse_expand
from test_pse import overlapping_text_example


def draw_title(img, title):
    output = cv2.resize(img, (500, 280), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(output, (0, 0), (500, 38), (24, 24, 24), -1)
    cv2.putText(output, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.70, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/psenet_step6/synthetic_demo.png')
    args = parser.parse_args()
    masks = overlapping_text_example()
    labels, stages = pse_expand(masks, return_steps=True)
    panels = [draw_title(cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR),
                         f'K{i+1} binary mask') for i, m in enumerate(masks)]
    panels += [draw_title(colorize_labels(stages[i]), f'Stage {i+1}: instance IDs')
               for i in range(3)]
    result = np.concatenate([
        np.concatenate(panels[:3], axis=1),
        np.concatenate(panels[3:], axis=1),
    ], axis=0)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), result):
        raise RuntimeError(f'Failed to save {path}')
    print(f'Instance IDs: {np.unique(labels[labels > 0]).tolist()}')
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
