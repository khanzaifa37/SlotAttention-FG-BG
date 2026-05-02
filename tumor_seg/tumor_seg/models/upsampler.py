from torch import nn


class Upsampler(nn.Module):
    """4x ConvTranspose decoder: 14 -> 28 -> 56 -> 112 -> 224 (patch_size=16)."""

    def __init__(self, in_channels: int = 256, out_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)
