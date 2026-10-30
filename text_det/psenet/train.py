"""LearnOCR PSENet Step 5: small, understandable training pipeline.

Run from LearnOCR repository root:
  python text_detection/psenet/train.py --mode overfit --name img_1.jpg --size 320
  python text_detection/psenet/train.py --mode train --pretrained --size 512 --epochs 20
"""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ICDAR2015Dataset, MEAN, STD, list_training_images
from loss import PSENetLoss
from model import MiniPSENet


def choose_device(requested):
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pixel_text_dice(logits, targets, valid):
    """Thresholded PIXEL Dice, not instance-level detection F1."""
    with torch.no_grad():
        p = (logits[:, -1].sigmoid() >= 0.5) & (valid > 0.5)
        g = (targets[:, -1] > 0.5) & (valid > 0.5)
        intersection = (p & g).sum(dim=(1, 2)).float()
        denominator = p.sum(dim=(1, 2)).float() + g.sum(dim=(1, 2)).float()
        return ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean().item()


def save_preview(model, sample, destination, device):
    """Visualize the same image: GT binary kernels vs predicted probabilities."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(sample["image"].unsqueeze(0).to(device))
        probs = logits.sigmoid()[0].float().cpu().numpy()
    model.train(was_training)

    normalized = sample["image"].cpu().numpy().transpose(1, 2, 0)
    rgb = np.uint8(np.clip((normalized * STD + MEAN) * 255, 0, 255))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    kernels = sample["kernels"].cpu().numpy()
    size = bgr.shape[0]

    def panel(im, label):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (min(size, 270), 30), (18, 18, 18), -1)
        cv2.putText(im, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.57, (255, 255, 255), 1, cv2.LINE_AA)
        return im

    top = [panel(bgr, "Input image")]
    bottom = [panel(bgr, "Input image")]
    for i in range(3):
        gt = np.uint8(kernels[i] * 255)
        top.append(panel(cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR), f"GT K{i + 1}"))
        prob = np.uint8(np.clip(probs[i] * 255, 0, 255))
        colored = cv2.applyColorMap(prob, cv2.COLORMAP_TURBO)
        bottom.append(panel(colored, f"Pred K{i + 1} (prob)"))
    canvas = np.concatenate([
        np.concatenate(top, axis=1),
        np.concatenate(bottom, axis=1),
    ], axis=0)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)


def save_checkpoint(path, model, optimizer, args, *, step=0, epoch=0,
                    best_value=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step, "epoch": epoch, "best_value": best_value,
        "model_config": {"num_kernels": 3, "fpn_channels": 64},
        # pathlib.Path is not supported by torch.load(weights_only=True).
        "train_config": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
    }, path)


def make_loaders(args):
    names = list_training_images(args.data)
    if args.mode == "overfit":
        if args.name not in names:
            raise FileNotFoundError(f"{args.name} not found in {args.data}/train_images")
        dataset = ICDAR2015Dataset(args.data, [args.name], size=args.size,
                                   augment=False)
        return dataset, None, {"overfit_image": args.name}

    if not 0 < args.val_fraction < 0.5:
        raise ValueError("val-fraction must be in (0,0.5)")
    if len(names) < 2:
        raise ValueError("Need at least two images for train/validation split")
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(names)).tolist()
    n_val = max(1, round(len(names) * args.val_fraction))
    val_names = [names[i] for i in indices[:n_val]]
    train_names = [names[i] for i in indices[n_val:]]
    train_ds = ICDAR2015Dataset(args.data, train_names, args.size, augment=True)
    val_ds = ICDAR2015Dataset(args.data, val_names, args.size, augment=False)
    return train_ds, val_ds, {"train_images": train_names,
                              "validation_images": val_names}


def move_batch(batch, device):
    return (batch["image"].to(device), batch["kernels"].to(device),
            batch["training_mask"].to(device))


def write_csv(writer, *, step, epoch, phase, loss, loss_text, loss_kernel, dice):
    writer.writerow({
        "step": step, "epoch": epoch, "phase": phase,
        "loss": f"{loss:.6f}", "text_loss": f"{loss_text:.6f}",
        "kernel_loss": f"{loss_kernel:.6f}", "pixel_dice": f"{dice:.6f}",
    })


def main():
    p = argparse.ArgumentParser(description="LearnOCR Minimal PSENet training")
    p.add_argument("--mode", choices=["overfit", "train"], default="overfit")
    p.add_argument("--data", default="../data/icdar2015")
    p.add_argument("--name", default="img_1.jpg", help="overfit image name")
    p.add_argument("--size", type=int, default=320,
                   help="square letterbox size, divisible by 32")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=2,
                   help="train mode only; overfit always batch size 1")
    p.add_argument("--overfit-steps", type=int, default=250)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", action="store_true",
                   help="load torchvision ImageNet ResNet18 weights (may download)")
    p.add_argument("--kernel-mask-mode", choices=["predicted", "gt"],
                   default="predicted", help="gt is diagnostic only, not paper-exact")
    p.add_argument("--save-every", type=int, default=50,
                   help="overfit visualization/ckpt interval")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--output", default=None,
                   help="default: outputs/psenet_step5/<mode>")
    p.add_argument("--resume", type=Path, default=None,
                   help="only load checkpoints generated by your own training")
    args = p.parse_args()

    if args.size < 64 or args.size % 32 != 0:
        p.error("--size must be >=64 and divisible by 32")
    if args.lr <= 0 or args.epochs < 1 or args.overfit_steps < 1:
        p.error("lr must be positive; epochs and overfit-steps must be >=1")
    if args.batch_size < 1 or args.num_workers < 0 or args.save_every < 1 or args.log_every < 1:
        p.error("batch-size, save-every, log-every must be >=1; num-workers >=0")

    seed_everything(args.seed)
    device = choose_device(args.device)
    out = Path(args.output or f"outputs/psenet_step5/{args.mode}")
    out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, split = make_loaders(args)
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Device={device}  mode={args.mode}  size={args.size}  "
          f"train={len(train_ds)}  val={len(val_ds) if val_ds else 0}", flush=True)

    # False on resume prevents an unnecessary ResNet weight download.
    model = MiniPSENet(num_kernels=3, pretrained=args.pretrained and args.resume is None)
    model = model.to(device)
    criterion = PSENetLoss(kernel_mask_mode=args.kernel_mask_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_step, start_epoch, best = 0, 0, float("inf")
    if args.resume:
        # WARNING: torch.load uses pickle in compatible PyTorch versions;
        # never load untrusted checkpoints.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        best = float(ckpt.get("best_value") or float("inf"))
        print(f"Resumed: step={start_step}, completed_epochs={start_epoch}")

    csv_path = out / "history.csv"
    append = bool(args.resume and csv_path.exists())
    with csv_path.open("a" if append else "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "step", "epoch", "phase", "loss", "text_loss", "kernel_loss", "pixel_dice"
        ])
        if not append:
            writer.writeheader()

        if args.mode == "overfit":
            sample = train_ds[0]
            if sample["kernels"][-1].sum().item() == 0:
                raise ValueError("Selected image has no valid text pixels at this size; "
                                 "choose a different --name or larger --size")
            image = sample["image"].unsqueeze(0).to(device)
            targets = sample["kernels"].unsqueeze(0).to(device)
            valid = sample["training_mask"].unsqueeze(0).to(device)
            if start_step == 0:
                save_preview(model, sample, out / "preview_step_0000.jpg", device)
            model.train()
            for step in range(start_step + 1, args.overfit_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                losses = criterion(logits, targets, valid)
                loss = losses["loss"]
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite loss on step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                dice = pixel_text_dice(logits.detach(), targets, valid)
                metrics = {k: float(v.detach().item()) for k, v in losses.items()}
                write_csv(writer, step=step, epoch=0, phase="overfit",
                          loss=metrics["loss"], loss_text=metrics["loss_text"],
                          loss_kernel=metrics["loss_kernel"], dice=dice)
                if step == 1 or step % args.log_every == 0 or step == args.overfit_steps:
                    print(f"step {step:4d}: loss={metrics['loss']:.4f} "
                          f"text={metrics['loss_text']:.4f} "
                          f"kernel={metrics['loss_kernel']:.4f} "
                          f"pixel_Dice={dice:.4f}", flush=True)
                if step % args.save_every == 0 or step == args.overfit_steps:
                    if metrics["loss"] < best:
                        best = metrics["loss"]
                        save_checkpoint(out / "best.pt", model, optimizer, args,
                                        step=step, best_value=best)
                    save_preview(model, sample, out / f"preview_step_{step:04d}.jpg",
                                 device)
                    save_checkpoint(out / "last.pt", model, optimizer, args,
                                    step=step, best_value=best)
                    csv_file.flush()
            print(f"Overfit complete; inspect previews and {csv_path}")
            return

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")

        step = start_step
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            train_total, train_dice, train_n = 0.0, 0.0, 0
            for batch in train_loader:
                images, targets, valid = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                losses = criterion(logits, targets, valid)
                if not torch.isfinite(losses["loss"]).item():
                    raise RuntimeError(f"Non-finite loss on epoch {epoch}")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                bs = images.shape[0]
                train_total += losses["loss"].item() * bs
                train_dice += pixel_text_dice(logits.detach(), targets, valid) * bs
                train_n += bs
                step += 1

            model.eval()
            val_total = val_text = val_kernel = val_dice = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    images, targets, valid = move_batch(batch, device)
                    logits = model(images)
                    losses = criterion(logits, targets, valid)
                    bs = images.shape[0]
                    val_total += losses["loss"].item() * bs
                    val_text += losses["loss_text"].item() * bs
                    val_kernel += losses["loss_kernel"].item() * bs
                    val_dice += pixel_text_dice(logits, targets, valid) * bs
                    val_n += bs

            mean_train = train_total / train_n
            mean_val = val_total / val_n
            mean_dice = val_dice / val_n
            write_csv(writer, step=step, epoch=epoch, phase="train",
                      loss=mean_train, loss_text=float("nan"),
                      loss_kernel=float("nan"), dice=train_dice / train_n)
            write_csv(writer, step=step, epoch=epoch, phase="val",
                      loss=mean_val, loss_text=val_text / val_n,
                      loss_kernel=val_kernel / val_n, dice=mean_dice)
            csv_file.flush()
            print(f"epoch {epoch:3d}/{args.epochs}: train_loss={mean_train:.4f} "
                  f"val_loss={mean_val:.4f} val_pixel_Dice={mean_dice:.4f}",
                  flush=True)
            save_checkpoint(out / "last.pt", model, optimizer, args,
                            step=step, epoch=epoch, best_value=min(best, mean_val))
            if mean_val < best:
                best = mean_val
                save_checkpoint(out / "best.pt", model, optimizer, args,
                                step=step, epoch=epoch, best_value=best)
            # Same fixed validation sample at every epoch for comparison.
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                save_preview(model, val_ds[0],
                             out / f"val_preview_epoch_{epoch:03d}.jpg", device)

    print(f"Training complete; checkpoints and previews: {out}")


if __name__ == "__main__":
    main()
