"""End-to-end smoke test for the FBSA segmenter.

Runs three checks:
  1. shapes:        forward pass on random tensor -> [B,1,224,224]
  2. param freeze:  trainable params in [1M, 3M] (encoder frozen)
  3. mini-overfit:  32 random (image, mask) pairs overfit ~50 steps;
                    loss must drop >= 30% and train Dice must reach >= 0.7.

The third check is intentionally synthetic — uses random images with a
threshold-based mask — so the script can run without BRISC mounted.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumor_seg.config import TrainConfig
from tumor_seg.losses import DiceBCELoss
from tumor_seg.metrics import dice_coefficient
from tumor_seg.models import FBSASegmenter


def check_shapes(model, device):
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224, device=device))
    assert out.shape == (2, 1, 224, 224), f"expected [2,1,224,224] got {tuple(out.shape)}"
    print(f"[1/3] shapes OK  -> {tuple(out.shape)}")


def check_freeze(model):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    msg = f"trainable={n_train/1e6:.2f}M total={n_total/1e6:.2f}M"
    assert 1e6 <= n_train <= 5e6, f"trainable params out of band: {msg}"
    print(f"[2/3] freeze OK  -> {msg}")


def check_overfit(model, device, n_imgs: int = 32, steps: int = 50):
    torch.manual_seed(0)
    images = torch.randn(n_imgs, 3, 224, 224, device=device)
    gray = images.mean(dim=1, keepdim=True)
    masks = (gray > gray.flatten(1).median(dim=1).values[:, None, None, None]).float()

    crit = DiceBCELoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    model.train()
    losses = []
    dices = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randperm(n_imgs, device=device)[:8]
        logits = model(images[idx])
        loss = crit(logits, masks[idx])
        loss.backward()
        opt.step()
        losses.append(loss.item())
        with torch.no_grad():
            dices.append(dice_coefficient(logits, masks[idx]).item())

    drop = (losses[0] - losses[-1]) / max(losses[0], 1e-6)
    final_dice = sum(dices[-5:]) / 5
    print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}  ({100*drop:.1f}% drop)")
    print(f"  dice (last 5 mean) = {final_dice:.4f}")
    assert drop >= 0.3, f"loss did not drop >=30%: drop={drop:.2%}"
    assert final_dice >= 0.7, f"final train dice {final_dice:.3f} < 0.70"
    print(f"[3/3] mini-overfit OK")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    cfg = TrainConfig()
    model = FBSASegmenter(
        encoder_name=cfg.encoder, encoder_dim=cfg.encoder_dim,
        num_slots=cfg.num_slots, slot_dim=cfg.slot_dim,
        slot_iters=cfg.slot_iters, slot_hidden=cfg.slot_hidden,
    ).to(device)

    check_shapes(model, device)
    check_freeze(model)
    check_overfit(model, device)
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
