"""Educational DBNet: ResNet18 + FPN + differentiable binarization.

Dependencies: torch, torchvision, numpy, opencv-python, shapely.
This is a didactic implementation, NOT pretrained and NOT a faithful
replica of the author's complete training pipeline.

Run: python dbnet_minimal.py
"""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from shapely.geometry import Polygon


def conv_head(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels // 4, 3, padding=1),
        nn.BatchNorm2d(channels // 4), nn.ReLU(inplace=True),
        nn.ConvTranspose2d(channels // 4, channels // 4, 2, 2),
        nn.BatchNorm2d(channels // 4), nn.ReLU(inplace=True),
        nn.ConvTranspose2d(channels // 4, 1, 2, 2),
        nn.Sigmoid(),
    )


class DBNet(nn.Module):
    def __init__(self, channels=128, k=50):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, channels, 1) for c in [64, 128, 256, 512]
        ])
        self.reduce = nn.ModuleList([
            nn.Conv2d(channels, channels // 4, 3, padding=1)
            for _ in range(4)
        ])
        self.prob_head = conv_head(channels)
        self.thresh_head = conv_head(channels)
        self.k = k

    def forward(self, x):
        h, w = x.shape[-2:]
        b = self.backbone
        y = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        c2 = b.layer1(y)   # stride 4
        c3 = b.layer2(c2)  # stride 8
        c4 = b.layer3(c3)  # stride 16
        c5 = b.layer4(c4)  # stride 32
        cs = [c2, c3, c4, c5]
        lat = [layer(t) for layer, t in zip(self.lateral, cs)]
        fused = [None] * 4
        fused[3] = lat[3]
        for i in (2, 1, 0):
            up = F.interpolate(fused[i + 1], size=lat[i].shape[-2:], mode='nearest')
            fused[i] = lat[i] + up
        target_size = fused[0].shape[-2:]
        fused = [F.interpolate(layer(t), size=target_size, mode='nearest')
                 for layer, t in zip(self.reduce, fused)]
        features = torch.cat(fused, dim=1)
        p = self.prob_head(features)
        p = F.interpolate(p, size=(h, w), mode='bilinear', align_corners=False)
        if not self.training:
            return p  # only P is required at inference
        t = self.thresh_head(features)
        t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
        approx_binary = torch.sigmoid(self.k * (p - t))
        return {'prob': p, 'thresh': t, 'approx_binary': approx_binary}


def balanced_bce(pred, gt, valid_mask, negative_ratio=3):
    """BCE with per-image online hard negative mining (OHEM)."""
    raw = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), gt, reduction='none')
    results = []
    for i in range(pred.shape[0]):
        pos = (gt[i] > 0.5) & (valid_mask[i] > 0.5)
        neg = (gt[i] <= 0.5) & (valid_mask[i] > 0.5)
        pos_losses = raw[i][pos]
        neg_losses = raw[i][neg]
        n = min(neg_losses.numel(), pos_losses.numel() * negative_ratio)
        if n == 0:
            # For empty-text images, train on some background pixels too.
            n = min(neg_losses.numel(), 256)
        top_neg = neg_losses.topk(n).values if n else raw.new_zeros((0,))
        results.append((pos_losses.sum() + top_neg.sum()) / max(pos_losses.numel() + n, 1))
    return torch.stack(results).mean()


def db_loss(pred, prob_gt, thresh_gt, thresh_mask, valid_mask):
    l_prob = balanced_bce(pred['prob'], prob_gt, valid_mask)
    l_binary = balanced_bce(pred['approx_binary'], prob_gt, valid_mask)
    l_thresh = (torch.abs(pred['thresh'] - thresh_gt) * thresh_mask).sum() / (thresh_mask.sum() + 1e-6)
    return l_prob + l_binary + 10.0 * l_thresh


def _rasterize(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly.is_empty:
        return mask
    geoms = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
    for g in geoms:
        if g.geom_type != 'Polygon' or g.is_empty:
            continue
        pts = np.round(np.asarray(g.exterior.coords)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def make_targets(polygons, h, w, shrink_ratio=0.4, t_min=0.3, t_max=0.7):
    """Make P and T targets for polygon annotations in input-image coordinates.
    Threshold labels use distance to original polygon segments, valid in expanded Gd.
    Shapely.buffer substitutes for the paper's Vatti/pyclipper offset.
    """
    p_gt = np.zeros((h, w), np.float32)
    t_raw = np.zeros((h, w), np.float32)
    t_mask = np.zeros((h, w), np.float32)
    for xy in polygons:
        poly = Polygon(np.asarray(xy, dtype=np.float64))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1 or poly.length < 1:
            continue
        d = poly.area * (1 - shrink_ratio**2) / poly.length
        shrink = poly.buffer(-d, join_style='round')
        expand = poly.buffer(d, join_style='round')
        p_gt = np.maximum(p_gt, _rasterize(shrink, h, w).astype(np.float32))
        expanded_mask = _rasterize(expand, h, w).astype(bool)
        t_mask[expanded_mask] = 1.

        yy, xx = np.where(expanded_mask)
        if not len(xx):
            continue
        pts = np.asarray(poly.exterior.coords)[:-1]
        nearest = np.full(xx.shape, np.inf, dtype=np.float64)
        xy_pixels = np.stack([xx, yy], axis=1).astype(np.float64)
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            edge = b - a
            frac = np.clip(((xy_pixels - a) @ edge) / (edge @ edge + 1e-10), 0, 1)
            nearest = np.minimum(nearest, np.linalg.norm(xy_pixels - (a + frac[:, None] * edge), axis=1))
        t_raw[yy, xx] = np.maximum(t_raw[yy, xx], 1 - np.clip(nearest / (d + 1e-6), 0, 1))
    t_gt = t_min + t_raw * (t_max - t_min)
    return (torch.from_numpy(p_gt)[None], torch.from_numpy(t_gt)[None],
            torch.from_numpy(t_mask)[None], torch.ones(1, h, w))


def postprocess(prob, pixel_thresh=0.3, box_thresh=0.6, unclip_ratio=1.5):
    """Return [([[x, y], ... 4 points], confidence), ...] for one HxW probability map."""
    prob = np.asarray(prob, dtype=np.float32)
    bitmap = (prob > pixel_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    h, w = prob.shape
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) < 3:
            continue
        rect = cv2.minAreaRect(contour)
        if min(rect[1]) < 3:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        score = float(cv2.mean(prob, mask)[0])
        if score < box_thresh:
            continue
        contour_poly = Polygon(contour[:, 0, :].astype(float))
        if not contour_poly.is_valid:
            contour_poly = contour_poly.buffer(0)
        if contour_poly.is_empty or contour_poly.length <= 0:
            continue
        d = contour_poly.area * unclip_ratio / contour_poly.length
        enlarged = contour_poly.buffer(d, join_style='round')
        if enlarged.is_empty or enlarged.geom_type != 'Polygon':
            continue
        coords = np.asarray(enlarged.exterior.coords, np.float32)
        bigrect = cv2.minAreaRect(coords)
        if min(bigrect[1]) < 5:
            continue
        box = cv2.boxPoints(bigrect)
        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)
        results.append((np.round(box).astype(int).tolist(), score))
    return results


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = DBNet(channels=64, k=50)
    x = torch.rand(1, 3, 128, 256)
    poly = np.array([[30, 40], [210, 35], [213, 71], [33, 75]])
    labels = [a.unsqueeze(0) for a in make_targets([poly], 128, 256)]
    net.train()
    output = net(x)
    loss = db_loss(output, *labels)
    loss.backward()
    print('train shapes:', {k: tuple(v.shape) for k, v in output.items()})
    print('training loss:', round(loss.item(), 4))
    net.eval()
    with torch.no_grad():
        p = net(x)[0, 0].cpu().numpy()
    boxes = postprocess(p)
    print('inference probability shape:', p.shape)
    print('predicted boxes (randomly initialized model):', len(boxes))
