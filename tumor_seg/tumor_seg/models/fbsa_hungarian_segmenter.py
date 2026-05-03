"""FBSA v4 — K-slot variant trained with Hungarian-matched per-slot CE loss.

Same encoder/projection/slot-attention/upsampler as v1, but with K=4
(configurable via cfg.num_slots) slots instead of K=2. Each slot is decoded
into its own full-resolution binary logit map by folding K into the batch
dimension and reusing the shared upsampler — so the decoder weights are
shared across slots, no extra params beyond the larger slot_attn.

Pair this arch with cfg.loss="hungarian" to train with HungarianCELoss
(see losses.HungarianCELoss). The loss matches the K predicted slot maps
against (1 FG + K-1 BG) target masks via Hungarian on a 1-IoU cost matrix
and applies per-slot BCE on the matched arrangement.

Inference: a `fg_slot_votes` buffer (incremented by HungarianCELoss in
training only) tracks which slot most often wins the FG match. forward()
returns that slot's logits as the [B,1,H,W] FG prediction so the existing
metrics / save paths in train.py keep working untouched. Until the buffer
is populated, slot 0 is used.

Kept as a standalone class so future variants can diverge freely.
"""

import math
from typing import Optional

import torch
from torch import nn

from .encoder import DinoEncoder
from .slot_attention import SlotAttention
from .upsampler import Upsampler


class FBSAHungarianSegmenter(nn.Module):
    def __init__(
        self,
        encoder_name: str = "dino_vits16",
        encoder_dim: int = 384,
        num_slots: int = 4,
        slot_dim: int = 256,
        slot_iters: int = 3,
        slot_hidden: int = 512,
        out_channels: int = 1,
        finetune_blocks_after: Optional[int] = None,
    ):
        super().__init__()
        assert num_slots >= 2, (
            f"fbsa_hungarian needs num_slots >= 2 (1 FG + >=1 BG slot); got {num_slots}. "
            f"Set cfg.num_slots=4 in TrainConfig."
        )
        assert out_channels == 1, "fbsa_hungarian is binary-segmentation only"

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.encoder = DinoEncoder(encoder_name, finetune_blocks_after=finetune_blocks_after)
        assert self.encoder.embed_dim == encoder_dim, (
            f"encoder_dim {encoder_dim} != actual {self.encoder.embed_dim}"
        )
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )
        self.slot_attn = SlotAttention(
            num_slots=num_slots, slot_dim=slot_dim,
            n_iters=slot_iters, hidden_dim=slot_hidden,
        )
        self.upsampler = Upsampler(in_channels=slot_dim, out_channels=1)

        # Per-slot FG win count, accumulated by HungarianCELoss during training.
        # Buffer (not param) so it persists in checkpoints and moves with .to().
        self.register_buffer("fg_slot_votes", torch.zeros(num_slots))

        # Side channel: stash the full [B, K, H, W] slot logits from the most
        # recent forward so HungarianCELoss can reach them without changing
        # forward()'s public contract ([B,1,H,W] always — keeps the existing
        # metrics / save paths in train.py untouched). Not an nn.Parameter or
        # nn.Module, so nn.Module.__setattr__ leaves it as a plain attribute.
        self._last_slot_logits: Optional[torch.Tensor] = None

    def fg_idx(self) -> int:
        """Index of the slot most often assigned to FG. Defaults to 0 until
        HungarianCELoss has populated fg_slot_votes."""
        if self.fg_slot_votes.sum() > 0:
            return int(self.fg_slot_votes.argmax().item())
        return 0

    def _compute_all_slot_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Decode every slot in parallel by folding K into the batch axis."""
        B = image.shape[0]
        tokens = self.encoder(image)              # [B, N, encoder_dim]
        tokens = self.proj(tokens)                # [B, N, D]

        slots, attn = self.slot_attn(tokens)      # [B, K, D], [B, N, K]
        K, D = self.num_slots, self.slot_dim
        N = tokens.shape[1]
        H = W = int(math.sqrt(N))

        # Per-slot spatial feature: attn[:,:,k:k+1] * slots[:,k:k+1,:]
        # broadcast to [B, K, N, D] in one shot.
        feat = attn.transpose(1, 2).unsqueeze(-1) * slots.unsqueeze(2)
        # Fold K into batch for parallel decode: [B*K, D, H, W]
        feat = feat.reshape(B * K, N, D).transpose(1, 2).reshape(B * K, D, H, W)

        slot_logits = self.upsampler(feat)        # [B*K, 1, H_full, W_full]
        H_full, W_full = slot_logits.shape[-2:]
        return slot_logits.reshape(B, K, H_full, W_full)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        slot_logits = self._compute_all_slot_logits(image)  # [B, K, H, W]
        self._last_slot_logits = slot_logits
        idx = self.fg_idx()
        return slot_logits[:, idx:idx + 1]                  # [B, 1, H, W]
