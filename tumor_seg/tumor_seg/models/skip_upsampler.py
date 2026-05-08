"""U-Net-style upsampler with concat skip connections from an ImageStem.

Replaces the plain ``Upsampler`` for v3+. Same 4x ConvTranspose backbone
(14 -> 28 -> 56 -> 112 -> 224) but at each intermediate scale we concatenate
the matching ImageStem feature map and run a 3x3 fuse-conv before the next
upsample. This gives the decoder real high-frequency boundary information
that the ViT's 14x14 grid cannot provide.
"""

from typing import Sequence

import torch
from torch import nn


class _UpFuse(nn.Module):
    """Upsample then fuse a skip via concat + 3x3 conv."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class SkipUpsampler(nn.Module):
    """4x ConvTranspose decoder with 3 skip points.

    in_channels  : feature channels at the 14x14 stage (e.g. 512 for v2 fusion).
    skip_channels: tuple of (skip_at_28, skip_at_56, skip_at_112) — matches
                   ImageStem.channels = (4*base, 2*base, base).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        out_channels: int = 1,
    ):
        super().__init__()
        s28, s56, s112 = skip_channels
        # 14 -> 28
        self.up1 = _UpFuse(in_ch=in_channels, out_ch=128, skip_ch=s28)
        # 28 -> 56
        self.up2 = _UpFuse(in_ch=128,        out_ch=64,  skip_ch=s56)
        # 56 -> 112
        self.up3 = _UpFuse(in_ch=64,         out_ch=32,  skip_ch=s112)
        # 112 -> 224 (no stem skip at this scale; final detail learned end-to-end)
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x, skips, return_features: bool = False):
        s28, s56, s112 = skips
        x = self.up1(x, s28)
        x = self.up2(x, s56)
        decoder_feat = x
        x = self.up3(x, s112)
        x = self.up4(x)
        logits = self.head(x)
        if return_features:
            return logits, decoder_feat
        return logits
