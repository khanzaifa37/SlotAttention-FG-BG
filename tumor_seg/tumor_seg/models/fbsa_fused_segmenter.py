"""FBSA v2 — slot signal fused with raw encoder features.

v1 (FBSASegmenter) feeds only ``fg_attn * fg_slot`` to the upsampler — a
single 256-d vector spatially modulated by a 1-channel attention map. The
upsampler has to hallucinate fine boundaries from a coarse heatmap, which
caps Dice around ~0.62 on BRISC.

v2 concatenates the projected encoder tokens onto that signal, doubling the
channels going into the upsampler so the decoder has real spatial features
to refine alongside the slot's "where is the tumor" hint.

Kept as a standalone class (not a subclass of FBSASegmenter) so the two
architectures can diverge freely as we iterate.
"""

import math
import torch
from torch import nn

from .encoder import DinoEncoder
from .slot_attention import SlotAttention
from .upsampler import Upsampler


class FBSAFusedSegmenter(nn.Module):
    def __init__(
        self,
        encoder_name: str = "dino_vits16",
        encoder_dim: int = 384,
        num_slots: int = 2,
        slot_dim: int = 256,
        slot_iters: int = 3,
        slot_hidden: int = 512,
        out_channels: int = 1,
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

        # Using attention mechanism, token keys and slot queries
        # update each slot as weighted sum of token values
        # After this the features are divided into 2 slots, 
        # 0=foreground (tumor), 1=background
        self.slot_attn = SlotAttention(
            num_slots=num_slots, slot_dim=slot_dim,
            n_iters=slot_iters, hidden_dim=slot_hidden,
        )
        # Upsampler input = slot_feat (slot_dim) + projected tokens (slot_dim)
        self.upsampler = Upsampler(in_channels=slot_dim * 2, out_channels=out_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        tokens = self.encoder(image)
        # Projecting the ViT patch tokens into slot attention dimensions
        # 256-dim feature vector, with LayerNorm
        tokens = self.proj(tokens)                   # [B, N, D]

        slots, attn = self.slot_attn(tokens)         # [B, K, D], [B, N, K]

        fg_slot = slots[:, 0:1, :]                   # [B, 1, D]
        fg_attn = attn[:, :, 0:1]                    # [B, N, 1]
        # Build a feature using FG slot vector and attention vector only
        slot_feat = fg_attn * fg_slot                # [B, N, D]
        
        # Along with attention features add the ViT Tokens to boost performance
        combined = torch.cat([slot_feat, tokens], dim=-1)  # [B, N, 2D]

        N = combined.shape[1]
        H = W = int(math.sqrt(N))
        feat = combined.transpose(1, 2).reshape(B, 2 * self.slot_dim, H, W)

        return self.upsampler(feat)
