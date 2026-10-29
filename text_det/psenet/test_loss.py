"""Tests for Step 4, no dataset required. Run from LearnOCR root."""

import torch

from loss import PSENetLoss, build_ohem_mask, masked_dice_loss
from model import MiniPSENet


def almost_equal(a, b, tol=1e-5):
    assert abs(float(torch.as_tensor(a).detach()) - float(torch.as_tensor(b).detach())) < tol, f"{a} != {b}"


def test_dice_and_ignore():
    gt = torch.zeros(1, 8, 8)
    gt[:, 2:6, 2:6] = 1
    valid = torch.ones_like(gt)
    perfect = torch.where(gt > 0, torch.tensor(1.0), torch.tensor(0.0))
    wrong = 1 - perfect
    almost_equal(masked_dice_loss(perfect, gt, valid), 0.0)
    assert masked_dice_loss(wrong, gt, valid).item() > 0.99

    # A false positive inside ignored area is completely invisible to the loss.
    valid[:, :2, :] = 0
    modified = perfect.clone()
    modified[:, :2, :] = 1
    almost_equal(masked_dice_loss(modified, gt, valid), 0.0)
    print("Dice + ignored pixels: PASS")


def test_ohem():
    target = torch.zeros(1, 4, 4)
    target[0, 0, 0] = 1
    probs = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 16
    valid = torch.ones_like(target)
    valid[0, 3, 3] = 0  # high probability but ignored
    mask = build_ohem_mask(probs, target, valid, negative_ratio=3)
    assert mask.sum().item() == 4  # 1 positive + 3 negatives
    assert mask[0, 0, 0].item() and not mask[0, 3, 3].item()
    assert mask[0, 3, 2].item() and mask[0, 3, 1].item() and mask[0, 3, 0].item()
    print("OHEM top-3 negatives + ignore mask: PASS")


def test_all_ignore_zero_loss():
    x = torch.randn(1, 3, 8, 8, requires_grad=True)
    gt = torch.zeros_like(x)
    valid = torch.zeros(1, 8, 8)
    output = PSENetLoss()(x, gt, valid)
    almost_equal(output["loss"], 0.0)
    output["loss"].backward()
    assert torch.all(x.grad == 0)
    print("All ignored gives loss=0 and gradient=0: PASS")




def test_ignored_logits_cannot_affect_loss():
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    targets = torch.zeros(1, 3, 8, 8)
    targets[:, 0, 3:5, 3:5] = 1
    targets[:, 1, 2:6, 2:6] = 1
    targets[:, 2, 1:7, 1:7] = 1
    valid = torch.ones(1, 8, 8)
    valid[:, :2, :] = 0
    criterion = PSENetLoss()
    before = criterion(logits, targets, valid)["loss"]
    edited = logits.detach().clone()
    edited[:, :, :2, :] = 99.0
    after = criterion(edited, targets, valid)["loss"]
    almost_equal(before, after)
    before.backward()
    assert torch.all(logits.grad[:, :, :2, :] == 0)
    print("Ignored logits don't change loss or gradient: PASS")


def test_kernel_mask_modes():
    targets = torch.zeros(1, 3, 8, 8)
    targets[:, 0, 3:5, 3:5] = 1
    targets[:, 1, 2:6, 2:6] = 1
    targets[:, 2, 1:7, 1:7] = 1
    logits = torch.full_like(targets, -5.0)  # predicted text < 0.5 everywhere
    valid = torch.ones(1, 8, 8)
    predicted = PSENetLoss(kernel_mask_mode="predicted")(logits, targets, valid)
    teacher = PSENetLoss(kernel_mask_mode="gt")(logits, targets, valid)
    almost_equal(predicted["loss_kernel"], 0.0)
    assert teacher["loss_kernel"].item() > 0.5
    print("Predicted-vs-GT kernel mask behavior: PASS")

def test_loss_backward():
    torch.manual_seed(0)
    model = MiniPSENet(num_kernels=3, pretrained=False)
    model.train()
    img = torch.rand(1, 3, 64, 64)
    gt = torch.zeros(1, 3, 64, 64)
    gt[:, 0, 22:37, 20:43] = 1
    gt[:, 1, 19:40, 17:46] = 1
    gt[:, 2, 16:44, 14:49] = 1
    valid = torch.ones(1, 64, 64)
    valid[:, :10, :10] = 0

    logits = model(img)
    result = PSENetLoss()(logits, gt, valid)
    assert result["loss"].isfinite()
    assert result["loss_text"].isfinite()
    assert result["loss_kernel"].isfinite()
    almost_equal(result["loss"],
                 .7 * result["loss_text"] + .3 * result["loss_kernel"])
    result["loss"].backward()
    for name, layer in (("ResNet stem", model.stem[0]),
                        ("FPN", model.lat2),
                        ("segmentation head", model.head[-1])):
        g = layer.weight.grad
        assert g is not None and g.isfinite().all(), f"No valid gradient: {name}"
    print("MiniPSENet + real PSENetLoss backward: PASS")
    print(f"  loss={result['loss'].item():.5f}, "
          f"text={result['loss_text'].item():.5f}, "
          f"kernel={result['loss_kernel'].item():.5f}")


if __name__ == "__main__":
    test_dice_and_ignore()
    test_ohem()
    test_all_ignore_zero_loss()
    test_ignored_logits_cannot_affect_loss()
    test_kernel_mask_modes()
    test_loss_backward()
    print("ALL STEP 4 TESTS PASSED")
