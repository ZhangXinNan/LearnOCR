"""Plot training history as two separate PNGs (requires matplotlib)."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="outputs/psenet_step5/overfit/history.csv")
    args = parser.parse_args()
    path = Path(args.history)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("history.csv is empty")
    output = path.parent
    for field, filename, ylabel in (
        ("loss", "loss_curve.png", "PSENet Loss"),
        ("pixel_dice", "pixel_dice_curve.png", "Thresholded Pixel Dice (K3)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for phase in sorted({row["phase"] for row in rows}):
            subset = [row for row in rows if row["phase"] == phase]
            x_col = "epoch" if phase != "overfit" else "step"
            xs = [int(row[x_col]) for row in subset]
            ys = [float(row[field]) for row in subset]
            ax.plot(xs, ys, marker="." if len(xs) < 30 else None, label=phase)
        ax.set_xlabel("Step" if rows[0]["phase"] == "overfit" else "Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)
        print(f"Saved {output / filename}")


if __name__ == "__main__":
    main()
