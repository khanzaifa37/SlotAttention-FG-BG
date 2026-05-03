import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .metrics import dice_coefficient


class DiceBCELoss(nn.Module):
    """Mirrors notebook cell 7. bce_weight=0.5 by default."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = 1.0 - dice_coefficient(logits, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


class HungarianCELoss(nn.Module):
    """Hungarian-matched per-slot CE loss for the K-slot binary segmenter.

    Adapted from contextfusion_bootstrp.py:hungarian_ce_loss, specialised for
    binary FG/BG ground truth. Per batch element:

      1. Build K target masks: index 0 = FG mask, indices 1..K-1 = BG mask
         (duplicated so the assignment matrix is square).
      2. Cost matrix [K, K] = 1 - IoU between each predicted slot's sigmoid
         probability map and each target mask.
      3. Hungarian solves the optimal assignment via scipy.linear_sum_assignment.
      4. Loss = sigmoid-BCE between each slot's logits and its assigned target.
      5. The slot that won the FG target is recorded as a vote on
         model.fg_slot_votes (training only — inference reads this buffer to
         pick which slot to expose as the FG prediction).

    Reads the full [B, K, H, W] slot logits via model._last_slot_logits, set
    by FBSAHungarianSegmenter.forward on every call. This avoids changing
    the model's public forward signature, so DiceBCELoss + the existing
    metric pipeline in train.py keep working untouched for the other arches.
    """

    def __init__(self, model: nn.Module, vote_momentum: float = 1.0):
        super().__init__()
        # Hold model as a single-element list so nn.Module.__setattr__ does
        # not auto-register it as a child module (which would double-count
        # its params under criterion.parameters() / criterion.to(device)).
        self._model_ref = [model]
        self.vote_momentum = vote_momentum

    @property
    def model(self) -> nn.Module:
        return self._model_ref[0]

    def forward(self, fg_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # fg_logits ([B,1,H,W]) is here only so the call signature matches
        # DiceBCELoss — the actual loss is computed on the full slot stack
        # stashed by the model's most recent forward() call.
        slot_logits = self.model._last_slot_logits
        if slot_logits is None:
            raise RuntimeError(
                "HungarianCELoss expected model._last_slot_logits to be set "
                "by FBSAHungarianSegmenter.forward(). Got None — check that "
                "cfg.arch is 'fbsa_hungarian' when cfg.loss is 'hungarian'."
            )

        B, K, H, W = slot_logits.shape
        fg = target.squeeze(1)            # [B, H, W]
        bg = 1.0 - fg                     # [B, H, W]

        total_loss = fg_logits.new_zeros(())
        for b in range(B):
            slots_prob = torch.sigmoid(slot_logits[b]).detach()  # cost only
            targets_b = torch.stack(
                [fg[b]] + [bg[b]] * (K - 1), dim=0
            )                                                    # [K, H, W]

            slots_flat = slots_prob.reshape(K, -1)
            targets_flat = targets_b.reshape(K, -1)
            inter = slots_flat @ targets_flat.T                  # [K, K]
            sum_s = slots_flat.sum(1, keepdim=True)
            sum_t = targets_flat.sum(1, keepdim=True).T
            union = sum_s + sum_t - inter + 1e-6
            cost = 1.0 - inter / union

            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            # Square matrix => row_ind is identity; col_ind[k] is the target
            # index assigned to slot k. Reorder targets to slot order.
            col_ind_t = torch.as_tensor(col_ind, device=targets_b.device, dtype=torch.long)
            aligned = targets_b.index_select(0, col_ind_t)       # [K, H, W]

            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                slot_logits[b], aligned
            )

            if self.model.training:
                # Slot that received target index 0 (FG) is the FG winner.
                fg_slot = int(np.where(col_ind == 0)[0][0])
                with torch.no_grad():
                    self.model.fg_slot_votes.mul_(self.vote_momentum)
                    self.model.fg_slot_votes[fg_slot] += 1.0

        return total_loss / B


def build_loss(cfg, model: nn.Module) -> nn.Module:
    """Pick a loss based on cfg.loss. Default 'dicebce' preserves prior behavior."""
    name = getattr(cfg, "loss", "dicebce")
    if name == "dicebce":
        return DiceBCELoss(bce_weight=cfg.bce_weight)
    if name == "hungarian":
        return HungarianCELoss(model)
    raise ValueError(f"unknown loss={name!r}; available: dicebce, hungarian")
