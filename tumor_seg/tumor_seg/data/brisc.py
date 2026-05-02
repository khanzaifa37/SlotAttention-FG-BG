"""BRISC-2025 segmentation dataset.

Mirrors the dataset class used in BRISC_Baseline_Colab.ipynb (cell 6) so that
metrics from this package are directly comparable to the notebook's UNet and
DINO baselines. Differences from the notebook:

- ``rgb_input=True`` is the default (DINO ViT requires 3 channels).
- ImageNet mean/std normalization is applied to RGB images so that DINO
  features land in the regime the encoder was pretrained on.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BriscSegmentationDataset(Dataset):
    def __init__(
        self,
        root,
        split: str,
        image_size: int = 224,
        augment: bool = False,
        rgb_input: bool = True,
        normalize: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.rgb_input = rgb_input
        self.normalize = normalize

        split_root = self.root / split
        self.images_dir = split_root / "images"
        self.masks_dir = split_root / "masks"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self.samples = []
        for image_path in sorted(self.images_dir.glob("*.jpg")):
            mask_path = self.masks_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                self.samples.append((image_path, mask_path))
        if not self.samples:
            raise RuntimeError(f"No image/mask pairs found in {split_root}")

    def __len__(self):
        return len(self.samples)

    def _load(self, image_path, mask_path):
        mode = "RGB" if self.rgb_input else "L"
        image = Image.open(image_path).convert(mode).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        return image, mask

    def _to_tensor(self, image, mask):
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        if self.rgb_input:
            image_t = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
            if self.normalize:
                image_t = TF.normalize(image_t, IMAGENET_MEAN, IMAGENET_STD)
        else:
            image_t = torch.from_numpy(image_np[None, ...]).float()

        mask_np = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_np[None, ...]).float()
        return image_t, mask_t

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image, mask = self._load(image_path, mask_path)
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image); mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image); mask = TF.vflip(mask)
        return self._to_tensor(image, mask)


def create_brisc_dataloaders(
    data_root,
    batch_size: int = 16,
    image_size: int = 224,
    train_limit=None,
    val_limit=None,
    augment_train: bool = True,
    rgb_input: bool = True,
    num_workers: int = 2,
):
    train_set = BriscSegmentationDataset(
        data_root, split="train", image_size=image_size,
        augment=augment_train, rgb_input=rgb_input,
    )
    val_set = BriscSegmentationDataset(
        data_root, split="test", image_size=image_size,
        augment=False, rgb_input=rgb_input,
    )
    if train_limit is not None:
        train_set = Subset(train_set, range(min(train_limit, len(train_set))))
    if val_limit is not None:
        val_set = Subset(val_set, range(min(val_limit, len(val_set))))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader
