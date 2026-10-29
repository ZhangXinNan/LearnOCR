

from pathlib import Path

import cv2
import numpy as np

ROOT = Path("../data/icdar2015")
image_path = ROOT / "train_images/img_1.jpg"
label_path = ROOT / "train_gt/gt_img_1.txt"

# 忽略 EXIF 自动旋转，保持标注与图像坐标一致
image = cv2.imread(
    str(image_path),
    cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
)
if image is None:
    raise FileNotFoundError(image_path)

num_boxes = 0

with open(label_path, "r", encoding="utf-8-sig") as f:
    for line in f:
        parts = line.strip().split(",", 8)
        if len(parts) < 9:
            continue

        coords = [int(float(v)) for v in parts[:8]]
        polygon = np.array(coords, dtype=np.int32).reshape(4, 2)
        transcription = parts[8].strip()

        # 绿色：有效文字；红色：忽略区域
        color = (0, 0, 255) if transcription == "###" else (0, 255, 0)
        cv2.polylines(image, [polygon], True, color, 2)
        num_boxes += 1

output_path = Path("outputs/gt_example.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output_path), image)
print(f"Image shape: {image.shape}")
print(f"Text boxes: {num_boxes}")
print(f"Saved to: {output_path}")



