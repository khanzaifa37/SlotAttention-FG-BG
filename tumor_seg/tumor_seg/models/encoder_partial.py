"""DINO ViT encoder with partial fine-tuning of the top transformer blocks.

This is the v4 counterpart to the fully-frozen ``DinoEncoder``. Patch embed,
positional encoding, CLS token, and the lower transformer blocks stay
frozen; only blocks with index ``>= finetune_blocks_after`` are trainable.
This matches the convention in the upstream ContextFusion repo
(``FB_Indicator.py``).

Why a separate class instead of parameterising ``DinoEncoder``: the existing
class hard-codes ``@torch.no_grad()`` on its forward (correct for v1-v3 since
nothing inside it needs gradients). v4 needs gradients to flow through the
trainable top blocks, so the forward must be grad-aware. Keeping the two
classes separate is cleaner than threading conditional grad-no_grad through
the existing path.
"""

import torch
from torch import nn


class DinoEncoderPartial(nn.Module):
    """DINO ViT with the top ``finetune_blocks_after``+ blocks left trainable.

    For DINO ViT-S/16 (12 blocks), ``finetune_blocks_after=10`` means blocks
    10 and 11 train; blocks 0..9 plus the patch embed / pos embed / CLS / norm
    stay frozen.
    """

    def __init__(self, name: str = "dino_vits16", finetune_blocks_after: int = 10):
        super().__init__()
        self.name = name
        self.vit = torch.hub.load("facebookresearch/dino:main", name)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size
        self.finetune_blocks_after = finetune_blocks_after

        for param_name, param in self.vit.named_parameters():
            if param_name.startswith("blocks."):
                block_id = int(param_name.split(".")[1])
                param.requires_grad = block_id >= finetune_blocks_after
            else:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embed + CLS + pos embed (frozen, but still need to flow through).
        with torch.no_grad():
            x = self.vit.prepare_tokens(x)
            for blk in self.vit.blocks[: self.finetune_blocks_after]:
                x = blk(x)

        # Top blocks: gradient-tracked.
        for blk in self.vit.blocks[self.finetune_blocks_after :]:
            x = blk(x)

        return x[:, 1:]
