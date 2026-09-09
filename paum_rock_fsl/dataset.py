# coding=utf-8
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class NJURockPairDataset(data.Dataset):
    """NJU rock thin-section split dataset for ProtoNet+CAFF+FS.

    The original CAFF code expects a 6-channel tensor formed by concatenating
    two RGB views. For each image whose filename ends with ``...-k`` where
    ``k > 1``, this dataset pairs it with the corresponding ``...-1`` image in
    the same class folder. Images ending in ``...-1`` are used as references
    and are not sampled directly.
    """

    def __init__(self, root: str | Path, split: str, transform=None):
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.transform = transform

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        class_dirs = sorted([p for p in self.split_dir.iterdir() if p.is_dir()])
        if not class_dirs:
            raise RuntimeError(f"No class folders found under {self.split_dir}")

        self.classes = [p.name for p in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: List[Tuple[Path, Path, int]] = []

        skipped_missing_pair = 0
        for class_dir in class_dirs:
            files = sorted(
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            by_stem: Dict[str, Path] = {p.stem: p for p in files}
            label = self.class_to_idx[class_dir.name]

            for image_path in files:
                pair_stems, view_idx = self._reference_stems(image_path.stem)
                if view_idx <= 1:
                    continue
                reference_path = None
                for pair_stem in pair_stems:
                    reference_path = by_stem.get(pair_stem)
                    if reference_path is not None:
                        break
                if reference_path is None:
                    skipped_missing_pair += 1
                    continue
                self.samples.append((image_path, reference_path, label))

        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.split_dir}")

        self.y = [label for _, _, label in self.samples]
        self.targets = self.y
        self.skipped_missing_pair = skipped_missing_pair

    @staticmethod
    def _reference_stems(stem: str) -> Tuple[List[str], int]:
        match = re.match(r"^(.*-)(\d+)(_.+)?$", stem)
        if not match:
            return [stem], 2
        prefix, view_idx, suffix = match.group(1), int(match.group(2)), match.group(3) or ""
        refs = [prefix + "1" + suffix]
        if suffix:
            refs.append(prefix + "1")
        return refs, view_idx

    def __getitem__(self, index):
        image_path, reference_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        reference = Image.open(reference_path).convert("RGB")
        view_idx = self._reference_stems(image_path.stem)[1]
        reference = reference.rotate(float((view_idx - 2) * 15))

        if self.transform is not None:
            image = self.transform(image)
            reference = self.transform(reference)

        return torch.cat([image, reference], dim=0), label

    def __len__(self):
        return len(self.samples)
