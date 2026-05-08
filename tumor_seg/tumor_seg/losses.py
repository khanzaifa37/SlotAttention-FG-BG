import torch
from torch import nn
import torch.nn.functional as F

from .metrics import dice_coefficient


class DiceBCELoss(nn.Module):
    """Mirrors notebook cell 7. bce_weight=0.5 by default."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, dict):
            logits = logits["logits"]
        bce_loss = self.bce(logits, targets)
        dice_loss = 1.0 - dice_coefficient(logits, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss over normalized embeddings."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        n = embeddings.shape[0]
        if n <= 1:
            return embeddings.sum() * 0.0

        labels = labels.reshape(-1)
        if labels.unique().numel() < 2:
            return embeddings.sum() * 0.0
        sim = embeddings @ embeddings.t() / self.temperature
        logits_mask = ~torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = labels[:, None].eq(labels[None, :]) & logits_mask

        valid = positive_mask.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0

        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim) * logits_mask.float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-8))
        mean_log_prob = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
        return -mean_log_prob[valid].mean()


class ContrastiveDiceBCELoss(nn.Module):
    """Dice+BCE segmentation loss plus mask-guided contrastive terms."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        lambda_token: float = 0.05,
        lambda_pixel: float = 0.05,
        lambda_slot: float = 0.02,
        temperature: float = 0.1,
        pixel_samples_fg: int = 128,
        pixel_samples_bg: int = 128,
    ):
        super().__init__()
        self.seg_loss = DiceBCELoss(bce_weight=bce_weight)
        self.supcon = SupervisedContrastiveLoss(temperature=temperature)
        self.lambda_token = lambda_token
        self.lambda_pixel = lambda_pixel
        self.lambda_slot = lambda_slot
        self.temperature = temperature
        self.pixel_samples_fg = pixel_samples_fg
        self.pixel_samples_bg = pixel_samples_bg
        self.last_components = {}

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if not isinstance(outputs, dict):
            loss = self.seg_loss(outputs, targets)
            self.last_components = {
                "seg_loss": loss.detach(),
                "token_contrast_loss": loss.detach() * 0.0,
                "pixel_contrast_loss": loss.detach() * 0.0,
                "slot_proto_loss": loss.detach() * 0.0,
                "total_loss": loss.detach(),
            }
            return loss

        seg = self.seg_loss(outputs["logits"], targets)
        token = self._token_contrast(outputs["token_emb"], targets)
        pixel = self._pixel_contrast(outputs["pixel_emb"], targets)
        slot = self._slot_proto(outputs["token_emb"], outputs["slot_emb"], targets)
        total = (
            seg
            + self.lambda_token * token
            + self.lambda_pixel * pixel
            + self.lambda_slot * slot
        )
        self.last_components = {
            "seg_loss": seg.detach(),
            "token_contrast_loss": token.detach(),
            "pixel_contrast_loss": pixel.detach(),
            "slot_proto_loss": slot.detach(),
            "total_loss": total.detach(),
        }
        return total

    def _token_contrast(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        b, n, d = tokens.shape
        h = w = int(n ** 0.5)
        labels = F.interpolate(targets.float(), size=(h, w), mode="nearest")
        labels = labels.reshape(b, -1).long()
        return self.supcon(tokens.reshape(b * n, d), labels.reshape(-1))

    def _pixel_contrast(self, pixel_feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pixel_feats.shape
        labels = F.interpolate(targets.float(), size=(h, w), mode="nearest").reshape(b, -1).long()
        feats = pixel_feats.permute(0, 2, 3, 1).reshape(b, h * w, c)

        sampled_feats = []
        sampled_labels = []
        for i in range(b):
            fg_idx = torch.nonzero(labels[i] == 1, as_tuple=False).flatten()
            bg_idx = torch.nonzero(labels[i] == 0, as_tuple=False).flatten()
            if fg_idx.numel() > self.pixel_samples_fg:
                fg_idx = fg_idx[torch.randperm(fg_idx.numel(), device=fg_idx.device)[:self.pixel_samples_fg]]
            if bg_idx.numel() > self.pixel_samples_bg:
                bg_idx = bg_idx[torch.randperm(bg_idx.numel(), device=bg_idx.device)[:self.pixel_samples_bg]]
            idx = torch.cat([fg_idx, bg_idx], dim=0)
            if idx.numel() == 0:
                continue
            sampled_feats.append(feats[i, idx])
            sampled_labels.append(labels[i, idx])

        if not sampled_feats:
            return pixel_feats.sum() * 0.0
        return self.supcon(torch.cat(sampled_feats, dim=0), torch.cat(sampled_labels, dim=0))

    def _slot_proto(
        self,
        tokens: torch.Tensor,
        slots: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = tokens.shape
        h = w = int(n ** 0.5)
        masks = F.interpolate(targets.float(), size=(h, w), mode="nearest").reshape(b, n, 1)
        fg_count = masks.sum(dim=1)
        bg_weight = 1.0 - masks
        bg_count = bg_weight.sum(dim=1)
        valid = (fg_count.squeeze(-1) > 0) & (bg_count.squeeze(-1) > 0)
        if not valid.any():
            return tokens.sum() * 0.0

        fg_proto = (tokens * masks).sum(dim=1) / fg_count.clamp_min(1.0)
        bg_proto = (tokens * bg_weight).sum(dim=1) / bg_count.clamp_min(1.0)
        protos = F.normalize(torch.stack([fg_proto, bg_proto], dim=1), dim=-1)

        slot_logits = torch.einsum("bkd,bpd->bkp", slots, protos) / self.temperature
        slot_logits = slot_logits[valid].reshape(-1, 2)
        labels = torch.tensor([0, 1], device=tokens.device).repeat(int(valid.sum().item()))
        return F.cross_entropy(slot_logits, labels)
