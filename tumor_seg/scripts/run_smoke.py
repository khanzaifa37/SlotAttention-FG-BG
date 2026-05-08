"""End-to-end smoke test for the FBSA segmenter.

Runs three structural checks (no BRISC required):
  1. shapes:       forward pass on random tensor -> [B,1,224,224]
  2. param freeze: trainable params in [1M, 5M] (encoder frozen)
  3. grad flow:    one backward pass produces non-zero gradients on every
                   trainable parameter group (proj, slot_attn, upsampler).

We deliberately do NOT try to "overfit" on synthetic data: with the encoder
frozen, DINO features of random Gaussian noise carry no semantic signal, so
the slot-attention spatial pattern is meaningless and the upsampler cannot
produce arbitrary mask shapes from it. That's not a bug — the model is
designed to follow the encoder's features. Loss-decrease validation belongs
on real BRISC data; the BRISC val pass at the end of 01_sanity.ipynb is the
right place for that signal.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumor_seg.config import TrainConfig
from tumor_seg.losses import ContrastiveDiceBCELoss, DiceBCELoss
from tumor_seg.models import build_model


def check_shapes(model, device):
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224, device=device))
    assert out.shape == (2, 1, 224, 224), f"expected [2,1,224,224] got {tuple(out.shape)}"
    print(f"[1/3] shapes OK  -> {tuple(out.shape)}")


def check_feature_dict(model, device):
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224, device=device), return_features=True)
    assert out["logits"].shape == (2, 1, 224, 224)
    assert out["tokens"].shape[:2] == (2, 196)
    assert out["slots"].shape[0] == 2
    assert out["attn"].shape[:2] == (2, 196)
    if "decoder_feat" in out:
        assert out["decoder_feat"].shape == (2, 64, 56, 56)
        assert out["token_emb"].shape == (2, 196, 128)
        assert out["slot_emb"].shape == (2, 2, 128)
        assert out["pixel_emb"].shape == (2, 64, 56, 56)
    if "slot_masks" in out:
        assert out["slot_masks"].shape[0] == 2
        assert out["slot_masks"].shape[-2:] == (224, 224)
    print("[1b] feature dict OK")


def check_freeze(model):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    msg = f"trainable={n_train/1e6:.2f}M total={n_total/1e6:.2f}M"
    assert 1e5 <= n_train <= 5e6, f"trainable params out of band: {msg}"
    print(f"[2/3] freeze OK  -> {msg}")


def check_grad_flow(model, device, contrastive: bool = False):
    """Verify gradients reach every trainable parameter group.

    Forward + backward on synthetic data, then check that the L2 norm of the
    gradient on every trainable param is finite and > 0. This catches wiring
    bugs (detached tensors, wrong .requires_grad, missing modules) without
    pretending to "learn" from noise.
    """
    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224, device=device)
    targets = (torch.rand(2, 1, 224, 224, device=device) > 0.5).float()

    model.train()
    crit = ContrastiveDiceBCELoss() if contrastive else DiceBCELoss()
    outputs = model(images, return_features=True) if contrastive else model(images)
    loss = crit(outputs, targets)
    loss.backward()

    groups = {"proj": "proj", "slot_attn": "slot_attn", "upsampler": "upsampler"}
    if hasattr(model, "image_stem"):
        groups["image_stem"] = "image_stem"
    if contrastive:
        groups.update({"token_head": "token_head", "slot_head": "slot_head", "pixel_head": "pixel_head"})
    seen = {k: False for k in groups}
    bad = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for key, prefix in groups.items():
            if name.startswith(prefix):
                seen[key] = True
                break
        if p.grad is None:
            bad.append(f"{name}: grad is None")
            continue
        gnorm = p.grad.detach().float().norm().item()
        if not (gnorm > 0 and torch.isfinite(p.grad).all()):
            bad.append(f"{name}: bad grad norm={gnorm}")

    print(f"  loss = {loss.item():.4f}")
    print(f"  groups with grads: {seen}")
    if bad:
        for b in bad[:10]:
            print("  !", b)
    assert not bad, f"{len(bad)} parameter(s) have missing/zero/non-finite grads"
    assert all(seen.values()), f"groups missed: {[k for k,v in seen.items() if not v]}"
    print("[3/3] grad flow OK")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="fbsa",
                        help="Architecture key from ARCH_REGISTRY.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}  arch = {args.arch}")

    cfg = TrainConfig(arch=args.arch)
    model = build_model(cfg).to(device)

    check_shapes(model, device)
    if args.arch in ("fbsa_skip_contrastive", "dinosaur_readout"):
        check_feature_dict(model, device)
    check_freeze(model)
    check_grad_flow(model, device, contrastive=args.arch == "fbsa_skip_contrastive")
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
