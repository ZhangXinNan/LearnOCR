"""Minimal, educational PSENet: ResNet18 + tiny FPN + 3-channel kernel head.

Output is raw logits [B, K, H, W], ordered K1 (smallest) -> K3 (full).
This is intentionally NOT an exact reproduction of the original PSENet network.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MiniPSENet(nn.Module):
    def __init__(self, num_kernels: int = 3, fpn_channels: int = 64,
                 pretrained: bool = False):
        super().__init__()
        if num_kernels < 2:
            raise ValueError("num_kernels must be >= 2 for progressive expansion")
        if fpn_channels <= 0:
            raise ValueError("fpn_channels must be positive")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # ResNet18 extracts 4 resolutions (C2-C5, strides 4/8/16/32).
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # C2:  64 channels, stride 4
        self.layer2 = backbone.layer2  # C3: 128 channels, stride 8
        self.layer3 = backbone.layer3  # C4: 256 channels, stride 16
        self.layer4 = backbone.layer4  # C5: 512 channels, stride 32

        # 1x1 lateral convolutions unify channel counts before addition.
        self.lat2 = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(128, fpn_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lat5 = nn.Conv2d(512, fpn_channels, kernel_size=1)

        # Combine all FPN scales at stride 4 and predict K kernel logits.
        self.head = nn.Sequential(
            nn.Conv2d(4 * fpn_channels, fpn_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, num_kernels, kernel_size=1),
        )

    @staticmethod
    def _up_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Resize to exact reference size (works for odd input dimensions)."""
        return F.interpolate(
            x, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected RGB image tensor [B, 3, H, W]")
        input_size = x.shape[-2:]

        # 1) Backbone: low-level detail -> high-level semantics.
        c2 = self.layer1(self.stem(x))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 2) Feature pyramid: propagate semantic features top-down.
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + self._up_to(p5, c4)
        p3 = self.lat3(c3) + self._up_to(p4, c3)
        p2 = self.lat2(c2) + self._up_to(p3, c2)

        # 3) Concat multi-scale features and predict kernel logits.
        fused = torch.cat(
            [p2, self._up_to(p3, p2), self._up_to(p4, p2),
             self._up_to(p5, p2)], dim=1
        )
        small_logits = self.head(fused)
        logits = F.interpolate(
            small_logits, size=input_size, mode="bilinear", align_corners=False
        )

        if return_features:
            return logits, {
                "C2": c2, "C3": c3, "C4": c4, "C5": c5,
                "P2": p2, "P3": p3, "P4": p4, "P5": p5,
                "fused": fused, "small_logits": small_logits,
            }
        return logits
