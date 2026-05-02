"""Baseline segmentation models from BRISC_Baseline_Colab.ipynb.

These were originally defined inline in the notebook. Lifting them into a
module lets 03_eval_viz.ipynb load their checkpoints alongside the FBSA
variants without duplicating class definitions.

Two architectures:
- UNet: classic 1-channel grayscale UNet (the notebook's first baseline)
- DinoViTSegmentationModel: frozen-DINO ViT encoder + 4x ConvTranspose decoder
  (the notebook's second baseline)

Both expect 224x224 input and produce 1-channel logits at the same resolution.
Preprocessing differs — see ``BASELINE_PREPROCESS`` in 03_eval_viz.ipynb.
"""

import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------- UNet ---------------------------------------- #


class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Classic UNet. Defaults match the BRISC baseline notebook (1 ch in / out)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.in_conv = _DoubleConv(in_channels, base_channels)
        self.down1 = _Down(base_channels, base_channels * 2)
        self.down2 = _Down(base_channels * 2, base_channels * 4)
        self.down3 = _Down(base_channels * 4, base_channels * 8)
        self.bottleneck = _Down(base_channels * 8, base_channels * 16)
        self.up1 = _Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = _Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = _Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = _Up(base_channels * 2, base_channels, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)


# --------------------- DINO ViT + ConvTranspose decoder -------------------- #


class DinoViTSegmentationModel(nn.Module):
    """Frozen DINO ViT-S/16 encoder + 4x ConvTranspose decoder.

    The encoder is provided externally (loaded via ``timm.create_model`` in the
    notebook). Decoder upsamples 14x14 -> 224x224 progressively.
    """

    def __init__(self, encoder, out_channels: int = 1, image_size: int = 224):
        super().__init__()
        self.encoder = encoder
        self.out_channels = out_channels
        self.image_size = image_size

        self.encoder_feature_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.encoder_feature_map_size = image_size // self.patch_size

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_feature_dim, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.encoder.forward_features(x)
        patch_tokens = x[:, 1:]
        B, N, D = patch_tokens.shape
        H = W = self.encoder_feature_map_size
        feat = patch_tokens.transpose(1, 2).reshape(B, D, H, W)
        return self.decoder(feat)


def build_dino_baseline(image_size: int = 224, out_channels: int = 1):
    """Convenience constructor used by both the training notebook and 03.

    Imports timm lazily so this module stays usable in environments without it
    (e.g., running just the UNet baseline).
    """
    import timm

    encoder = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return DinoViTSegmentationModel(encoder=encoder, out_channels=out_channels, image_size=image_size)
