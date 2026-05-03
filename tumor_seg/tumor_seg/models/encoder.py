import torch
from torch import nn
from typing import Optional


class DinoEncoder(nn.Module):
    """DINO ViT-S/16 encoder.

    Frozen by default. Set finetune_blocks_after=N to unfreeze all ViT blocks
    with index >= N (0-indexed). E.g. finetune_blocks_after=10 unfreezes the
    last 2 blocks of a 12-block ViT-S/16, which is the recommended setting for
    adapting DINO features to MRI/tumor images.

    Frozen prefix blocks run under torch.no_grad() during forward so they
    don't materialise an autograd graph, keeping memory overhead low.
    """

    def __init__(self, name: str = "dino_vits16", finetune_blocks_after: Optional[int] = None):
        super().__init__()
        self.name = name
        self.finetune_blocks_after = finetune_blocks_after
        self.vit = torch.hub.load("facebookresearch/dino:main", name)
        self.vit.eval()

        for p in self.vit.parameters():
            p.requires_grad = False

        if finetune_blocks_after is not None:
            for param_name, param in self.vit.named_parameters():
                if "blocks" in param_name:
                    block_id = int(param_name.split(".")[1])
                    if block_id >= finetune_blocks_after:
                        param.requires_grad = True

        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.finetune_blocks_after is None:
            with torch.no_grad():
                x = self.vit.prepare_tokens(x)
                for blk in self.vit.blocks:
                    x = blk(x)
                return x[:, 1:]

        with torch.no_grad():
            x = self.vit.prepare_tokens(x)

        for i, blk in enumerate(self.vit.blocks):
            if i >= self.finetune_blocks_after:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)

        return x[:, 1:]
