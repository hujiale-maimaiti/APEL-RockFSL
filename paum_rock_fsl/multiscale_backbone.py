from __future__ import annotations

from typing import NamedTuple

import torch

from .resnet_backbone import FGKMiniResNet18Backbone


class MultiScaleFeatureMaps(NamedTuple):
    layer3: torch.Tensor
    layer4: torch.Tensor


class FGKMultiScaleResNet18Backbone(FGKMiniResNet18Backbone):
    """FGK ResNet18 with an opt-in multi-scale feature-map interface.

    The default ``forward`` path remains identical to the frozen baseline.
    PAUM-RockFSL uses ``forward_multiscale`` to retain layer3 local detail while
    preserving the original layer4 map consumed by TDPF and CUPM.
    """

    layer3_channels = 256
    layer4_channels = 512

    def forward_multiscale(self, x: torch.Tensor) -> MultiScaleFeatureMaps:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        layer3 = self.layer3(x)
        layer4 = self.layer4(layer3)
        return MultiScaleFeatureMaps(layer3=layer3, layer4=layer4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_multiscale(x).layer4
