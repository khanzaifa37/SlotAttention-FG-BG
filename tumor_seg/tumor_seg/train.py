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
from .losses import ContrastiveDiceBCELoss, DiceBCELoss
from .metrics import dice_coefficient, iou_coefficient, hausdorff_distance_metric
from .models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_model(model, images, return_features: bool = False):
    if return_features:
        try:
            return model(images, return_features=True)
        except TypeError:
            return model(images)
    return model(images)


def _component_value(criterion, name: str, loss: torch.Tensor) -> float:
    components = getattr(criterion, "last_components", None) or {}
    value = components.get(name)
    if value is None:
        if name == "seg_loss":
            return float(loss.item())
        return 0.0
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def _best_slot_metrics(outputs, masks, image_size):
    if not isinstance(outputs, dict) or "slot_masks" not in outputs:
        return None

    with torch.no_grad():
        slot_probs = outputs["slot_masks"].detach().clamp(1e-6, 1.0 - 1e-6)
        targets = masks.detach()
        inter = (slot_probs * targets).flatten(2).sum(dim=2)
        sums = slot_probs.flatten(2).sum(dim=2) + targets.flatten(1).sum(dim=1, keepdim=True)
        dice_per_slot = (2 * inter + 1e-6) / (sums + 1e-6)
        best_idx = dice_per_slot.argmax(dim=1)
        chosen = slot_probs[torch.arange(slot_probs.shape[0], device=slot_probs.device), best_idx].unsqueeze(1)
        chosen_logits = torch.logit(chosen)
        return {
            "oracle_dice": dice_coefficient(chosen_logits, masks).item(),
            "oracle_iou": iou_coefficient(chosen_logits, masks).item(),
            "oracle_hd": hausdorff_distance_metric(chosen_logits, masks, image_size=image_size),
        }


def run_epoch(model, loader, criterion, device, image_size, optimizer=None, return_features: bool = False):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_dice = total_iou = total_hd = 0.0
    total_seg_loss = total_token_loss = total_pixel_loss = total_slot_loss = 0.0
    total_oracle_dice = total_oracle_iou = total_oracle_hd = 0.0
    has_oracle = False
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = _forward_model(model, images, return_features=return_features)
            loss = criterion(outputs, masks)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            dice = dice_coefficient(logits, masks)
            iou = iou_coefficient(logits, masks)
            hd = hausdorff_distance_metric(logits, masks, image_size=image_size)
            oracle = _best_slot_metrics(outputs, masks, image_size)

        if is_train:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_dice += dice.item() * bs
        total_iou += iou.item() * bs
        total_hd += hd * bs
        total_seg_loss += _component_value(criterion, "seg_loss", loss) * bs
        total_token_loss += _component_value(criterion, "token_contrast_loss", loss) * bs
        total_pixel_loss += _component_value(criterion, "pixel_contrast_loss", loss) * bs
        total_slot_loss += _component_value(criterion, "slot_proto_loss", loss) * bs
        if oracle is not None:
            has_oracle = True
            total_oracle_dice += oracle["oracle_dice"] * bs
            total_oracle_iou += oracle["oracle_iou"] * bs
            total_oracle_hd += oracle["oracle_hd"] * bs

    n = len(loader.dataset)
    stats = {
        "loss": total_loss / n,
        "seg_loss": total_seg_loss / n,
        "token_contrast_loss": total_token_loss / n,
        "pixel_contrast_loss": total_pixel_loss / n,
        "slot_proto_loss": total_slot_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "hd": total_hd / n,
    }
    if has_oracle:
        stats.update({
            "oracle_dice": total_oracle_dice / n,
            "oracle_iou": total_oracle_iou / n,
            "oracle_hd": total_oracle_hd / n,
        })
    return stats


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

    use_contrastive = cfg.contrastive_enabled or cfg.arch == "fbsa_skip_contrastive"
    use_feature_outputs = use_contrastive or cfg.return_features or cfg.arch.startswith("dinosaur")
    if use_contrastive:
        criterion = ContrastiveDiceBCELoss(
            bce_weight=cfg.bce_weight,
            lambda_token=cfg.lambda_token,
            lambda_pixel=cfg.lambda_pixel,
            lambda_slot=cfg.lambda_slot,
            temperature=cfg.contrastive_temperature,
            pixel_samples_fg=cfg.pixel_samples_fg,
            pixel_samples_bg=cfg.pixel_samples_bg,
        )
    else:
        criterion = DiceBCELoss(bce_weight=cfg.bce_weight)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    best_dice = 0.0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_stats = run_epoch(
            model, train_loader, criterion, device, cfg.image_size, optimizer,
            return_features=use_feature_outputs,
        )
        val_stats = run_epoch(
            model, val_loader, criterion, device, cfg.image_size,
            return_features=use_feature_outputs,
        )
        scheduler.step()
        row = dict(
            epoch=epoch, lr=optimizer.param_groups[0]["lr"],
            train_loss=train_stats["loss"],
            train_seg_loss=train_stats["seg_loss"],
            train_token_contrast_loss=train_stats["token_contrast_loss"],
            train_pixel_contrast_loss=train_stats["pixel_contrast_loss"],
            train_slot_proto_loss=train_stats["slot_proto_loss"],
            train_dice=train_stats["dice"],
            train_iou=train_stats["iou"],
            train_hd=train_stats["hd"],
            val_loss=val_stats["loss"],
            val_seg_loss=val_stats["seg_loss"],
            val_token_contrast_loss=val_stats["token_contrast_loss"],
            val_pixel_contrast_loss=val_stats["pixel_contrast_loss"],
            val_slot_proto_loss=val_stats["slot_proto_loss"],
            val_dice=val_stats["dice"],
            val_iou=val_stats["iou"],
            val_hd=val_stats["hd"],
        )
        for key in ("oracle_dice", "oracle_iou", "oracle_hd"):
            if key in train_stats:
                row[f"train_{key}"] = train_stats[key]
            if key in val_stats:
                row[f"val_{key}"] = val_stats[key]
        history.append(row)
        print(row)

        if val_stats["dice"] > best_dice:
            best_dice = val_stats["dice"]
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
        if f.name in ("train_limit", "val_limit"):
            kw["type"] = int
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
