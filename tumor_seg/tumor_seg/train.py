"""Training entrypoint for the FBSA tumor segmenter.

Usage:
    python -m tumor_seg.train --data_root /path/to/brisc/segmentation_task

Mirrors the run_epoch / metric structure of BRISC_Baseline_Colab.ipynb so the
loss curves and tables are directly comparable.
"""

import argparse
import math
import os
import random
from dataclasses import asdict

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import TrainConfig
from .data import create_brisc_dataloaders
from .losses import build_loss
from .metrics import dice_coefficient, iou_coefficient, hausdorff_distance_metric
from .models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, image_size, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_dice = total_iou = total_hd = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_coefficient(logits, masks)
            iou = iou_coefficient(logits, masks)
            hd = hausdorff_distance_metric(logits, masks, image_size=image_size)

        if is_train:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_dice += dice.item() * bs
        total_iou += iou.item() * bs
        total_hd += hd * bs

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n, total_hd / n


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_loader, val_loader = create_brisc_dataloaders(
        cfg.data_root,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        train_limit=cfg.train_limit,
        val_limit=cfg.val_limit,
        num_workers=cfg.num_workers,
    )
    print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    model = build_model(cfg).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"arch={cfg.arch}  params trainable={n_train/1e6:.2f}M  total={n_total/1e6:.2f}M")

    criterion = build_loss(cfg, model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    best_dice = 0.0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_dice, tr_iou, tr_hd = run_epoch(
            model, train_loader, criterion, device, cfg.image_size, optimizer
        )
        val_loss, val_dice, val_iou, val_hd = run_epoch(
            model, val_loader, criterion, device, cfg.image_size
        )
        scheduler.step()
        row = dict(
            epoch=epoch, lr=optimizer.param_groups[0]["lr"],
            train_loss=tr_loss, train_dice=tr_dice, train_iou=tr_iou, train_hd=tr_hd,
            val_loss=val_loss, val_dice=val_dice, val_iou=val_iou, val_hd=val_hd,
        )
        history.append(row)
        print(row)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": asdict(cfg)},
                os.path.join(cfg.out_dir, "best_model.pt"),
            )
            print(f"  ↑ saved best (val_dice={best_dice:.4f})")

    torch.save({"history": history}, os.path.join(cfg.out_dir, "history.pt"))
    return history


def _build_parser():
    p = argparse.ArgumentParser("FBSA tumor segmenter")
    cfg = TrainConfig()
    for f in cfg.__dataclass_fields__.values():
        kw = {"type": type(f.default) if f.default is not None else str, "default": f.default}
        if f.type is bool or isinstance(f.default, bool):
            kw["type"] = lambda v: v.lower() in ("true", "1", "yes")
        p.add_argument(f"--{f.name}", **kw)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg = TrainConfig(**{k: v for k, v in vars(args).items() if v is not None})
    if not cfg.data_root:
        raise SystemExit("--data_root is required")
    main(cfg)
