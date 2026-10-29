"""Create two simple Step 7 charts from evaluation JSON/CSV.

Example:
    python text_detection/psenet/plot_evaluation.py \
      --dir outputs/psenet_step7/validation
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='outputs/psenet_step7/validation')
    args = parser.parse_args()
    folder = Path(args.dir)
    data = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    with (folder / 'per_image.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = ('precision', 'recall', 'f1')
    values = [data[k] for k in keys]
    ax.bar(['Precision', 'Recall', 'F1'], values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Micro-averaged instance score')
    ax.set_title(f'{data["split"].upper()} — {data["images"]} images')
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
    fig.tight_layout()
    fig.savefig(folder / 'metrics.png', dpi=150)
    plt.close(fig)

    # Do not show misleading empty-GT case as a meaningful per-image F1.
    f1_values = [float(row['f1']) for row in rows if int(row['num_care_gt']) > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(f1_values, bins=10, range=(0, 1))
    ax.set_xlabel('Per-image F1 (images with at least one care GT)')
    ax.set_ylabel('Number of images')
    ax.set_title('Detection F1 distribution')
    fig.tight_layout()
    fig.savefig(folder / 'f1_distribution.png', dpi=150)
    plt.close(fig)
    print('Saved:', folder / 'metrics.png', 'and', folder / 'f1_distribution.png')


if __name__ == '__main__':
    main()
