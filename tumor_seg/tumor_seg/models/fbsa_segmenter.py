import math
from typing import Optional

import torch
from torch import nn

from .encoder import DinoEncoder
from .slot_attention import SlotAttention
from .upsampler import Upsampler


class FBSASegmenter(nn.Module):
    """Foreground/Background Slot Attention segmenter.

    Frozen DINO ViT-S/16 -> linear proj to slot_dim -> 2-slot Slot Attention ->
    spatial broadcast of the FG slot (slot 0) weighted by its attention map ->
    ConvTranspose upsampler -> per-pixel logits.
    """

    def __init__(
        self,
        encoder_name: str = "dino_vits16",
        encoder_dim: int = 384,
        num_slots: int = 2,
        slot_dim: int = 256,
        slot_iters: int = 3,
        slot_hidden: int = 512,
        out_channels: int = 1,
        finetune_blocks_after: Optional[int] = None,
    ):
        super().__init__()
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
        self.upsampler = Upsampler(in_channels=slot_dim, out_channels=out_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        tokens = self.encoder(image)
        tokens = self.proj(tokens)

        slots, attn = self.slot_attn(tokens)

        fg_slot = slots[:, 0:1, :]
        fg_attn = attn[:, :, 0:1]
        feat = fg_attn * fg_slot

        N = feat.shape[1]
        H = W = int(math.sqrt(N))
        feat = feat.transpose(1, 2).reshape(B, self.slot_dim, H, W)

        return self.upsampler(feat)
