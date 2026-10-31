"""Shared PSENet inference: load weights -> predict kernels -> PSE -> boxes.

Both predict.py (one image) and evaluate.py (dataset) call detect_text().
The *generic* preprocessing and device selection live in preprocess.py/utils.py.
"""
import torch

from model import MiniPSENet
from preprocess import bgr_to_tensor, letterbox, to_original
from pse import binarize_kernels, extract_instances, pse_expand


def load_model(checkpoint, device):
    """Load old and new LearnOCR checkpoints without restoring an optimizer."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt.get("model_config", {})
    model = MiniPSENet(num_kernels=int(cfg.get("num_kernels", 3)),
                       fpn_channels=int(cfg.get("fpn_channels", 64)),
                       pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    input_size = int(ckpt.get("train_config", {}).get("size", 512))
    return model, input_size


def detect_text(model, image, size, device, *, threshold=0.5,
                min_kernel_area=5, min_area=30, min_score=0.5,
                debug=False):
    """Return detected instances in ORIGINAL image coordinates.

    debug=True also returns intermediate masks and PSE expansion stages.
    """
    canvas, valid, transform = letterbox(image, size)
    x = bgr_to_tensor(canvas, batched=True).to(device)
    with torch.inference_mode():
        probabilities = model(x).sigmoid()[0].float().cpu().numpy()
    probabilities[:, ~valid] = 0.0
    masks = binarize_kernels(probabilities, threshold)
    result = pse_expand(masks, min_kernel_area=min_kernel_area, return_steps=debug)
    if debug:
        labels, stages = result
    else:
        labels, stages = result, None
    instances = extract_instances(labels, probabilities[-1], min_area=min_area,
                                  min_score=min_score)
    for det in instances:
        det["box"] = to_original(det["box"], transform)
        det["contour"] = to_original(det["contour"], transform)
    details = {"canvas": canvas, "masks": masks, "labels": labels,
               "steps": stages} if debug else None
    return instances, details
