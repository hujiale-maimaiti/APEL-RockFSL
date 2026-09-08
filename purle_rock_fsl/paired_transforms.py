from __future__ import annotations

from typing import Sequence, Tuple

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


class SynchronizedPairTransform:
    """Apply one geometric transform to both views in a paired sample."""

    def __init__(
        self,
        train: bool,
        size: int = 224,
        resize_size: int = 256,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        horizontal_flip_probability: float = 0.5,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.train = bool(train)
        self.size = int(size)
        self.resize_size = int(resize_size)
        self.scale = tuple(float(value) for value in scale)
        self.ratio = tuple(float(value) for value in ratio)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)

    @staticmethod
    def _align_canvas(left: Image.Image, right: Image.Image):
        if right.size != left.size:
            right = TF.resize(
                right,
                [left.height, left.width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        return left, right

    def _train_geometry(self, left: Image.Image, right: Image.Image):
        top, left_offset, height, width = RandomResizedCrop.get_params(
            left,
            scale=self.scale,
            ratio=self.ratio,
        )
        crop_args = (
            top,
            left_offset,
            height,
            width,
            [self.size, self.size],
            InterpolationMode.BILINEAR,
        )
        left = TF.resized_crop(left, *crop_args, antialias=True)
        right = TF.resized_crop(right, *crop_args, antialias=True)
        if torch.rand(()).item() < self.horizontal_flip_probability:
            left = TF.hflip(left)
            right = TF.hflip(right)
        return left, right

    def _evaluation_geometry(self, left: Image.Image, right: Image.Image):
        left = TF.resize(
            left,
            self.resize_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        right = TF.resize(
            right,
            self.resize_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        left = TF.center_crop(left, [self.size, self.size])
        right = TF.center_crop(right, [self.size, self.size])
        return left, right

    def __call__(self, left: Image.Image, right: Image.Image):
        left, right = self._align_canvas(left, right)
        if self.train:
            left, right = self._train_geometry(left, right)
        else:
            left, right = self._evaluation_geometry(left, right)
        left = TF.normalize(TF.to_tensor(left), self.mean, self.std)
        right = TF.normalize(TF.to_tensor(right), self.mean, self.std)
        return left, right
