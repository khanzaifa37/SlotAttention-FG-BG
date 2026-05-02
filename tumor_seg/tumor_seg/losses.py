import torch
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
