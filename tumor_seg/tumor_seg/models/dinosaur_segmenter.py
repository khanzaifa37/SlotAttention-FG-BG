"""DINOSAUR-style Slot Attention baselines for tumor segmentation.

This is a vanilla object-centric baseline: frozen DINO tokens are projected,
grouped by Slot Attention, and decoded without the FBSA foreground slot rule
or image-stem skip connections. The supervised readout predicts a binary mask
from the full slot mixture, while optional returned slot masks support
best-slot/oracle evaluation of the object-centric attention maps.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import DinoEncoder
from .slot_attention import SlotAttention
from .upsampler import Upsampler


class DinosaurReadoutSegmenter(nn.Module):
    def __init__(
        self,
        encoder_name: str = "dino_vits16",
        encoder_dim: int = 384,
        num_slots: int = 4,
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
        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            n_iters=slot_iters,
            hidden_dim=slot_hidden,
        )
        self.upsampler = Upsampler(in_channels=slot_dim, out_channels=out_channels)

    def forward(self, image: torch.Tensor, return_features: bool = False):
        b = image.shape[0]
        tokens = self.proj(self.encoder(image))
        slots, attn = self.slot_attn(tokens)

        slot_mixture = torch.einsum("bnk,bkd->bnd", attn, slots)
        n = slot_mixture.shape[1]
        h = w = int(math.sqrt(n))
        feat = slot_mixture.transpose(1, 2).reshape(b, self.slot_dim, h, w)
        logits = self.upsampler(feat)

        if return_features:
            slot_masks = attn.transpose(1, 2).reshape(b, self.num_slots, h, w)
            slot_masks = F.interpolate(
                slot_masks,
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return {
                "logits": logits,
                "tokens": tokens,
                "slots": slots,
                "attn": attn,
                "slot_masks": slot_masks,
            }

        return logits
