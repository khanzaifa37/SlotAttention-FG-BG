"""FBSA v4 - FBSA-Skip with contrastive feature heads.

The segmentation path is intentionally the same as ``FBSASkipSegmenter``:
frozen DINO tokens, projected tokens, two-slot attention, token/slot fusion,
and the image-stem skip decoder. This variant additionally exposes projected
token embeddings, slots, attention maps, and an intermediate decoder feature
map so training can add mask-guided contrastive objectives.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import DinoEncoder
from .image_stem import ImageStem
from .skip_upsampler import SkipUpsampler
from .slot_attention import SlotAttention


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class _PixelProjectionHead(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class FBSASkipContrastiveSegmenter(nn.Module):
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
        token_embed_dim: int = 128,
        pixel_embed_dim: int = 64,
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
        self.upsampler = SkipUpsampler(
            in_channels=slot_dim * 2,
            skip_channels=self.image_stem.channels,
            out_channels=out_channels,
        )
        self.token_head = _ProjectionHead(slot_dim, out_dim=token_embed_dim)
        self.slot_head = _ProjectionHead(slot_dim, out_dim=token_embed_dim)
        self.pixel_head = _PixelProjectionHead(64, out_dim=pixel_embed_dim)

    def forward(self, image: torch.Tensor, return_features: bool = None):
        if return_features is None:
            return_features = self.training

        B = image.shape[0]
        skips = self.image_stem(image)

        tokens = self.encoder(image)
        tokens = self.proj(tokens)

        slots, attn = self.slot_attn(tokens)

        fg_slot = slots[:, 0:1, :]
        fg_attn = attn[:, :, 0:1]
        slot_feat = fg_attn * fg_slot

        combined = torch.cat([slot_feat, tokens], dim=-1)
        N = combined.shape[1]
        H = W = int(math.sqrt(N))
        feat = combined.transpose(1, 2).reshape(B, 2 * self.slot_dim, H, W)

        if return_features:
            logits, decoder_feat = self.upsampler(feat, skips, return_features=True)
            return {
                "logits": logits,
                "tokens": tokens,
                "slots": slots,
                "attn": attn,
                "decoder_feat": decoder_feat,
                "token_emb": self.token_head(tokens),
                "slot_emb": self.slot_head(slots),
                "pixel_emb": self.pixel_head(decoder_feat),
            }

        return self.upsampler(feat, skips)
