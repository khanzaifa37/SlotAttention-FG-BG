"""FBSA v3 — v2 fusion + multi-scale image-stem skip connections (option 2).

Same encoder/projection/slot-attention as v2, and the same channel-wise
concat of ``fg_attn * fg_slot`` with the projected encoder tokens at the
14x14 stage. The new piece is an ImageStem that processes the raw image
into 28x28, 56x56, and 112x112 feature maps; a SkipUpsampler concatenates
those into the decoder stream U-Net style. This gives the decoder real
high-frequency information rather than relying entirely on the ViT's
coarse 14x14 grid for boundary recovery.

Kept as its own class (not a subclass of v2) so future variants can diverge
freely.
"""

import math
import torch
from torch import nn

from .encoder import DinoEncoder
from .image_stem import ImageStem
from .skip_upsampler import SkipUpsampler
from .slot_attention import SlotAttention


class FBSASkipSegmenter(nn.Module):
    def __init__(
        self,
        encoder_name: str = "dino_vits16",
        encoder_dim: int = 384,
        num_slots: int = 2,
        slot_dim: int = 256,
        slot_iters: int = 3,
        slot_hidden: int = 512,
        out_channels: int = 1,
        stem_base: int = 16,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.encoder = DinoEncoder(encoder_name)
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
        self.image_stem = ImageStem(in_channels=3, base=stem_base)
        # in_channels at 14x14 = slot_feat (slot_dim) + projected tokens (slot_dim)
        self.upsampler = SkipUpsampler(
            in_channels=slot_dim * 2,
            skip_channels=self.image_stem.channels,
            out_channels=out_channels,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        skips = self.image_stem(image)               # (s28, s56, s112)

        tokens = self.encoder(image)
        tokens = self.proj(tokens)                   # [B, N, D]

        slots, attn = self.slot_attn(tokens)         # [B, K, D], [B, N, K]

        fg_slot = slots[:, 0:1, :]
        fg_attn = attn[:, :, 0:1]
        slot_feat = fg_attn * fg_slot                # [B, N, D]

        combined = torch.cat([slot_feat, tokens], dim=-1)  # [B, N, 2D]
        N = combined.shape[1]
        H = W = int(math.sqrt(N))
        feat = combined.transpose(1, 2).reshape(B, 2 * self.slot_dim, H, W)

        return self.upsampler(feat, skips)
