from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .dataset import NJURockPairDataset


class NJURockSynchronizedPairDataset(NJURockPairDataset):
    """NJU paired dataset whose two views share geometric augmentation."""

    def __init__(self, root: str | Path, split: str, pair_transform):
        super().__init__(root=root, split=split, transform=None)
        self.pair_transform = pair_transform

    def __getitem__(self, index):
        image_path, reference_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        reference = Image.open(reference_path).convert("RGB")
        view_idx = self._reference_stems(image_path.stem)[1]
        reference = reference.rotate(float((view_idx - 2) * 15))

        image, reference = self.pair_transform(image, reference)
        return torch.cat([image, reference], dim=0), label
