# coding=utf-8
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision import models as tv_models


class FGKMiniResNet18Backbone(nn.Module):
    """ResNet18 feature-map backbone compatible with FGK ImageEncoder weights."""

    out_channels = 512

    def __init__(self):
        super().__init__()
        model = tv_models.resnet18(weights=None)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def extract_fgk_image_encoder_state(checkpoint):
    if isinstance(checkpoint, dict) and "image_encoder" in checkpoint:
        return checkpoint["image_encoder"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return OrderedDict(
            (key.replace("image_encoder.", "", 1), value)
            for key, value in checkpoint["model"].items()
            if key.startswith("image_encoder.")
        )
    return checkpoint


def load_fgk_resnet18_backbone(backbone, checkpoint, strict=False):
    image_encoder_state = extract_fgk_image_encoder_state(checkpoint)
    backbone_state = OrderedDict()
    for key, value in image_encoder_state.items():
        if key.startswith("backbone."):
            backbone_state[key.replace("backbone.", "", 1)] = value

    backbone.load_state_dict(backbone_state, strict=strict)
    return backbone
