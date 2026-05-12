"""Label-efficiency sweep for fbsa_skip.

Trains the fbsa_skip architecture at 10%, 25%, 50%, and 100% of the training
data, holding the full validation set constant so curves are directly
comparable.

Usage (programmatic):
    from tumor_seg.label_efficiency import run_label_efficiency
    results = run_label_efficiency(data_root="/path/to/brisc/segmentation_task",
                                   out_dir="/path/to/runs/fbsa_skip_label_eff")

Usage (CLI):
    python -m tumor_seg.label_efficiency \
        --data_root /path/to/brisc/segmentation_task \
        --out_dir   runs/fbsa_skip_label_eff
"""

import argparse
import os
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import torch

from .config import TrainConfig
from .data.brisc import BriscSegmentationDataset
from .train import main


FRACTIONS: tuple = (0.10, 0.25, 0.50, 1.00)


def run_label_efficiency(
    data_root: str,
    out_dir: str,
    fractions: Sequence[float] = FRACTIONS,
    base_cfg: Optional[TrainConfig] = None,
    skip_existing: bool = True,
) -> Dict[float, List[dict]]:
    """Train fbsa_skip at each data fraction and return all histories.

    Args:
        data_root:      Path to BRISC segmentation_task directory.
        out_dir:        Root output directory; each fraction gets a sub-folder
                        named ``frac_010``, ``frac_025``, etc.
        fractions:      Sequence of floats in (0, 1] to evaluate.
        base_cfg:       Optional TrainConfig to inherit hyperparameters from.
                        Defaults to standard fbsa_skip settings.
        skip_existing:  If True, skip a fraction whose ``history.pt`` already
                        exists and load its saved history instead.

    Returns:
        Dict mapping fraction (float) → history list.  Each history list is
        the same format returned by ``train.main``: a list of per-epoch dicts
        with keys ``epoch``, ``lr``, ``train_loss``, ``train_dice``,
        ``train_iou``, ``train_hd``, ``val_loss``, ``val_dice``, ``val_iou``,
        ``val_hd``.
    """
    if base_cfg is None:
        base_cfg = TrainConfig(arch="fbsa_skip", data_root=data_root, out_dir=out_dir)

    # Probe total training set size once without building full loaders.
    probe = BriscSegmentationDataset(
        data_root, split="train", image_size=base_cfg.image_size
    )
    n_total = len(probe)
    del probe

    print(f"Total training samples: {n_total}")
    print(f"Fractions to evaluate: {[f'{int(f*100)}%' for f in fractions]}\n")

    results: Dict[float, List[dict]] = {}
    for frac in fractions:
        pct = int(round(frac * 100))
        train_limit = max(base_cfg.batch_size, int(round(frac * n_total)))
        run_dir = os.path.join(out_dir, f"frac_{pct:03d}")
        history_path = os.path.join(run_dir, "history.pt")

        print(f"{'='*62}")
        print(f"  {pct:3d}%  —  {train_limit}/{n_total} training samples  →  {run_dir}")
        print(f"{'='*62}")

        if skip_existing and os.path.exists(history_path):
            print(f"  [skip] history.pt already exists — loading saved results.")
            results[frac] = torch.load(history_path, weights_only=False)["history"]
            continue

        cfg = replace(
            base_cfg,
            arch="fbsa_skip",
            data_root=data_root,
            out_dir=run_dir,
            train_limit=train_limit,
        )
        results[frac] = main(cfg)
        print()

    return results


def _best_row(history: List[dict]) -> dict:
    """Return the epoch dict with the highest val_dice."""
    return max(history, key=lambda r: r["val_dice"])


def summarise(results: Dict[float, List[dict]]) -> List[dict]:
    """Build a summary list (one row per fraction) for tabular display."""
    rows = []
    for frac in sorted(results):
        best = _best_row(results[frac])
        rows.append(
            {
                "fraction": frac,
                "pct": f"{int(round(frac*100))}%",
                "best_epoch": best["epoch"],
                "val_dice": best["val_dice"],
                "val_iou": best["val_iou"],
                "val_hd": best["val_hd"],
                "train_dice_at_best": best["train_dice"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Label-efficiency sweep for fbsa_skip")
    p.add_argument("--data_root", required=True, help="Path to BRISC segmentation_task")
    p.add_argument("--out_dir", default="runs/fbsa_skip_label_eff",
                   help="Root directory for all fraction sub-runs")
    p.add_argument("--fractions", nargs="+", type=float, default=list(FRACTIONS),
                   help="Data fractions to evaluate (e.g. 0.10 0.25 0.50 1.00)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_slots", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_skip_existing", action="store_true",
                   help="Re-train even if history.pt already exists")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    base = TrainConfig(
        arch="fbsa_skip",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_slots=args.num_slots,
        seed=args.seed,
        data_root=args.data_root,
        out_dir=args.out_dir,
    )
    results = run_label_efficiency(
        data_root=args.data_root,
        out_dir=args.out_dir,
        fractions=args.fractions,
        base_cfg=base,
        skip_existing=not args.no_skip_existing,
    )
    print("\n=== Summary ===")
    for row in summarise(results):
        print(
            f"  {row['pct']:>4s}  best_epoch={row['best_epoch']:2d}"
            f"  val_dice={row['val_dice']:.4f}"
            f"  val_iou={row['val_iou']:.4f}"
            f"  val_hd={row['val_hd']:.1f}px"
        )
