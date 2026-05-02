from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 30
    lr: float = 3e-4
    lr_min: float = 1e-6

    arch: str = "fbsa"          # see ARCH_REGISTRY in models/__init__.py
                                 # "fbsa"        — v1: slot signal only
                                 # "fbsa_fused"  — v2: slot + encoder feature fusion

    encoder: str = "dino_vits16"
    encoder_dim: int = 384
    patch_size: int = 16

    num_slots: int = 2
    slot_dim: int = 256
    slot_iters: int = 3
    slot_hidden: int = 512

    bce_weight: float = 0.5
    num_workers: int = 2
    train_limit: int = None
    val_limit: int = None

    data_root: str = ""
    out_dir: str = "runs/fbsa"
