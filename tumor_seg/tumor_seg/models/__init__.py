from .encoder import DinoEncoder
from .slot_attention import SlotAttention
from .upsampler import Upsampler
from .fbsa_segmenter import FBSASegmenter
from .fbsa_fused_segmenter import FBSAFusedSegmenter


# Registry of available architectures. Add new entries here as we add classes
# (e.g. fbsa_skip for option 2, fbsa_multi for option 4).
ARCH_REGISTRY = {
    "fbsa": FBSASegmenter,
    "fbsa_fused": FBSAFusedSegmenter,
}


def build_model(cfg):
    """Instantiate a segmenter from a TrainConfig.

    Selects the class via ``cfg.arch``. Common kwargs (encoder, slot dims)
    are forwarded to whichever class is selected — every architecture
    supported by the registry must accept the same constructor signature.
    """
    if cfg.arch not in ARCH_REGISTRY:
        raise ValueError(
            f"unknown arch={cfg.arch!r}; available: {list(ARCH_REGISTRY)}"
        )
    cls = ARCH_REGISTRY[cfg.arch]
    return cls(
        encoder_name=cfg.encoder,
        encoder_dim=cfg.encoder_dim,
        num_slots=cfg.num_slots,
        slot_dim=cfg.slot_dim,
        slot_iters=cfg.slot_iters,
        slot_hidden=cfg.slot_hidden,
    )
