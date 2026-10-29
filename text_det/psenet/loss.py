"""LearnOCR PSENet Step 4: square-Dice loss with ignore mask and OHEM.

Ordering is K1, K2, K3 (K3 == complete text region), as in Steps 2/3.
Run tests: python text_detection/psenet/test_loss.py
"""

import torch
from torch import nn


def masked_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Square-denominator Dice, averaged over batch (one Dice per image).

    pred, target, mask: [B, H, W], pred is a probability in [0,1].
    Ignored pixels make no contribution to the numerator or denominator.
    """
    if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 3:
        raise ValueError("pred, target, mask must all have shape [B, H, W]")
    mask = mask.to(dtype=pred.dtype)
    target = target.to(dtype=pred.dtype)
    p = (pred * mask).flatten(start_dim=1)
    g = (target * mask).flatten(start_dim=1)

    numerator = 2 * (p * g).sum(dim=1) + eps
    denominator = p.square().sum(dim=1) + g.square().sum(dim=1) + eps
    return (1 - numerator / denominator).mean()


@torch.no_grad()
def build_ohem_mask(text_probs: torch.Tensor, text_gt: torch.Tensor,
                    training_mask: torch.Tensor, negative_ratio: int = 3
                    ) -> torch.Tensor:
    """Keep all valid positive pixels + at most ratio*positive hard negatives.

    If an image has no positives, retain all valid negatives for a well-defined
    text loss. Masks are boolean and deliberately non-differentiable.
    """
    if text_probs.shape != text_gt.shape or text_probs.shape != training_mask.shape:
        raise ValueError("OHEM inputs must have identical shapes [B, H, W]")
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be > 0")

    selected = torch.zeros_like(text_gt, dtype=torch.bool)
    valid = training_mask > 0.5
    for b in range(text_gt.shape[0]):
        positive = (text_gt[b] > 0.5) & valid[b]
        negative = (text_gt[b] <= 0.5) & valid[b]
        num_pos = int(positive.sum().item())
        num_neg = int(negative.sum().item())
        if num_pos == 0:
            selected[b] = valid[b]
            continue
        k = min(negative_ratio * num_pos, num_neg)
        chosen = positive.reshape(-1).clone()
        if k > 0:
            neg_indices = torch.nonzero(negative.reshape(-1), as_tuple=False).flatten()
            scores = text_probs[b].detach().reshape(-1)[neg_indices]
            hard_indices = neg_indices[torch.topk(scores, k=k).indices]
            chosen[hard_indices] = True
        selected[b] = chosen.reshape_as(text_gt[b])
    return selected


class PSENetLoss(nn.Module):
    """Paper-style text + shrunk-kernel Dice loss for educational PSENet.

    Args:
        text_weight: lambda, default .7.
        use_ohem: whether to mine hard negatives for complete-text loss.
        kernel_mask_mode: 'predicted' (paper-style) or 'gt' (debug/warmup
            variant, NOT paper-exact).
    """

    def __init__(self, text_weight: float = 0.7, use_ohem: bool = True,
                 negative_ratio: int = 3,
                 kernel_mask_mode: str = "predicted"):
        super().__init__()
        if not 0 <= text_weight <= 1:
            raise ValueError("text_weight must be in [0, 1]")
        if negative_ratio <= 0:
            raise ValueError("negative_ratio must be > 0")
        if kernel_mask_mode not in ("predicted", "gt"):
            raise ValueError("kernel_mask_mode must be 'predicted' or 'gt'")
        self.text_weight = text_weight
        self.use_ohem = use_ohem
        self.negative_ratio = negative_ratio
        self.kernel_mask_mode = kernel_mask_mode

    def forward(self, logits: torch.Tensor, kernels: torch.Tensor,
                training_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.ndim != 4 or logits.shape[1] < 2:
            raise ValueError("logits must have shape [B, K>=2, H, W]")
        if logits.shape != kernels.shape:
            raise ValueError("logits and kernels must have the same shape")
        if training_mask.shape != (logits.shape[0], *logits.shape[-2:]):
            raise ValueError("training_mask must be [B, H, W]")
        probs = logits.sigmoid()
        gt = kernels.to(device=logits.device, dtype=logits.dtype)
        valid = (training_mask.to(device=logits.device) > 0.5)

        # Final channel is full text; all earlier channels are smaller kernels.
        text_probs, text_gt = probs[:, -1], gt[:, -1]
        text_mask = (build_ohem_mask(text_probs, text_gt, valid,
                     self.negative_ratio) if self.use_ohem else valid)
        loss_text = masked_dice_loss(text_probs, text_gt, text_mask)

        # W = predicted complete-text mask, intersected with valid pixels.
        # Mask selection is deliberately detached from the gradient graph.
        with torch.no_grad():
            if self.kernel_mask_mode == "predicted":
                kernel_mask = (text_probs.detach() >= 0.5) & valid
            else:
                kernel_mask = (text_gt > 0.5) & valid

        kernel_losses = [
            masked_dice_loss(probs[:, i], gt[:, i], kernel_mask)
            for i in range(logits.shape[1] - 1)
        ]
        loss_kernel = torch.stack(kernel_losses).mean()
        loss = self.text_weight * loss_text + (1 - self.text_weight) * loss_kernel
        return {"loss": loss, "loss_text": loss_text,
                "loss_kernel": loss_kernel}
