"""Read ICDAR2015 quadrilateral annotations (independent of PSENet/DBNet)."""
from pathlib import Path

import numpy as np


def read_icdar_annotations(path):
    """Return polygons as float32 [4,2], with a boolean ### ignore flag."""
    polygons, ignored = [], []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split(",", 8)
            if len(fields) != 9:
                raise ValueError(f"Invalid ICDAR annotation in {path}: {line!r}")
            coords = np.asarray([float(x) for x in fields[:8]], dtype=np.float32)
            polygons.append(coords.reshape(4, 2))
            ignored.append(fields[8].strip() == "###")
    return polygons, ignored
