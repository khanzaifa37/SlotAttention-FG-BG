import torch
from torch import nn


class DinoEncoder(nn.Module):
    """Frozen DINO ViT-S/16. Returns patch tokens [B, N, D] with CLS removed."""

    def __init__(self, name: str = "dino_vits16"):
        super().__init__()
        self.name = name
        self.vit = torch.hub.load("facebookresearch/dino:main", name)
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.prepare_tokens(x)
        for blk in self.vit.blocks:
            x = blk(x)
        return x[:, 1:]
