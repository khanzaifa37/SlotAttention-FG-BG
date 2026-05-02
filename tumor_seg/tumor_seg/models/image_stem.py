"""Tiny multi-scale CNN over the raw input image.

DINO ViT processes the image at a single 14x14 patch grid, so the upsampler
in v1/v2 has no fine-spatial signal to work with. ImageStem produces three
intermediate feature maps from the raw image — at 112, 56, and 28 — which
the SkipUpsampler concatenates in at the matching upsampling stages, U-Net
style. Roughly 50k params, all trainable.
"""

from torch import nn


def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class ImageStem(nn.Module):
    """3-stage CNN: 224 -> 112 -> 56 -> 28.

    Output channel widths default to (16, 32, 64). Returned in coarse-to-fine
    order so the SkipUpsampler can index from its first upsample step (which
    lands at 28) down to the last (which lands at 112).
    """

    def __init__(self, in_channels: int = 3, base: int = 16):
        super().__init__()
        self.s1 = _conv_block(in_channels, base, stride=2)        # 224 -> 112, base
        self.s2 = _conv_block(base, base * 2, stride=2)           # 112 -> 56,  2*base
        self.s3 = _conv_block(base * 2, base * 4, stride=2)       # 56  -> 28,  4*base
        self.channels = (base * 4, base * 2, base)                # (28, 56, 112) order

    def forward(self, x):
        s1 = self.s1(x)   # [B,  base, 112, 112]
        s2 = self.s2(s1)  # [B, 2base,  56,  56]
        s3 = self.s3(s2)  # [B, 4base,  28,  28]
        return s3, s2, s1
