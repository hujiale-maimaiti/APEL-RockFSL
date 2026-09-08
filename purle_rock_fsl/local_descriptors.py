from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .multiscale_backbone import MultiScaleFeatureMaps


class LocalDescriptorStreams(NamedTuple):
    common_map: torch.Tensor
    polarization_map: torch.Tensor
    common_tokens: torch.Tensor
    polarization_tokens: torch.Tensor
    fisher_score: torch.Tensor
    polarization_gate: torch.Tensor


class DualStreamLocalDescriptor(nn.Module):
    """Build morphology and signed polarization descriptors before pooling."""

    def __init__(
        self,
        layer3_channels: int = 256,
        layer4_channels: int = 512,
        descriptor_dim: int = 48,
        fisher_rank: int = 16,
        semantic_scale: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.layer4_channels = int(layer4_channels)
        self.descriptor_dim = int(descriptor_dim)
        self.semantic_scale = float(semantic_scale)
        self.eps = float(eps)

        # These projectors are shared by both views, keeping their descriptor
        # coordinates directly comparable.
        self.layer3_projection = nn.Conv2d(
            layer3_channels, descriptor_dim, kernel_size=1, bias=False
        )
        self.layer4_projection = nn.Conv2d(
            layer4_channels, descriptor_dim, kernel_size=1, bias=False
        )
        self.polarization_gate = nn.Sequential(
            nn.Linear(layer4_channels, fisher_rank),
            nn.GELU(),
            nn.Linear(fisher_rank, descriptor_dim),
        )
        nn.init.kaiming_normal_(
            self.layer3_projection.weight, mode="fan_out", nonlinearity="linear"
        )
        nn.init.kaiming_normal_(
            self.layer4_projection.weight, mode="fan_out", nonlinearity="linear"
        )
        nn.init.trunc_normal_(self.polarization_gate[0].weight, std=0.02)
        nn.init.zeros_(self.polarization_gate[0].bias)
        nn.init.zeros_(self.polarization_gate[-1].weight)
        nn.init.zeros_(self.polarization_gate[-1].bias)

    @staticmethod
    def _validate_support_indices(
        support_indices: Sequence[torch.Tensor], sample_count: int
    ):
        if not support_indices:
            raise ValueError("support_indices must contain at least one class")
        for index in support_indices:
            if index.ndim != 1 or index.numel() == 0:
                raise ValueError("each support index tensor must be non-empty and 1D")
            if int(index.min()) < 0 or int(index.max()) >= sample_count:
                raise IndexError("support index is outside the episode batch")

    def _task_fisher_score(
        self,
        ppl_layer4: torch.Tensor,
        xpl_layer4: torch.Tensor,
        support_indices: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        self._validate_support_indices(support_indices, ppl_layer4.size(0))
        ppl = F.adaptive_avg_pool2d(ppl_layer4, 1).flatten(1)
        xpl = F.adaptive_avg_pool2d(xpl_layer4, 1).flatten(1)
        paired_mean = 0.5 * (ppl + xpl)
        class_supports = [paired_mean[index] for index in support_indices]
        class_means = torch.stack([value.mean(0) for value in class_supports])
        within = torch.stack(
            [
                (value - mean.unsqueeze(0)).pow(2).mean(0)
                + 0.25 * (ppl[index] - xpl[index]).pow(2).mean(0)
                for value, mean, index in zip(
                    class_supports, class_means, support_indices
                )
            ]
        ).mean(0)
        between = (
            class_means - class_means.mean(0, keepdim=True)
        ).pow(2).mean(0)
        fisher = torch.log1p(between / (within + self.eps))
        return F.layer_norm(fisher, (self.layer4_channels,))

    def _project_view(self, features: MultiScaleFeatureMaps) -> torch.Tensor:
        local = self.layer3_projection(features.layer3)
        semantic = self.layer4_projection(features.layer4)
        semantic = F.interpolate(
            semantic,
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return local + self.semantic_scale * semantic

    @staticmethod
    def _tokens(feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(2).transpose(1, 2).contiguous()

    def forward(
        self,
        ppl_features: MultiScaleFeatureMaps,
        xpl_features: MultiScaleFeatureMaps,
        support_indices: Sequence[torch.Tensor],
    ) -> LocalDescriptorStreams:
        if ppl_features.layer3.shape != xpl_features.layer3.shape:
            raise ValueError("PPL/XPL layer3 feature maps must have equal shapes")
        if ppl_features.layer4.shape != xpl_features.layer4.shape:
            raise ValueError("PPL/XPL layer4 feature maps must have equal shapes")

        fisher = self._task_fisher_score(
            ppl_features.layer4,
            xpl_features.layer4,
            support_indices,
        )
        # A neutral initialization of one preserves the signed difference.
        gate = 1.0 + 0.5 * torch.tanh(self.polarization_gate(fisher))
        gate_map = gate.view(1, -1, 1, 1)

        ppl_local = self._project_view(ppl_features)
        xpl_local = self._project_view(xpl_features)
        common = F.normalize(0.5 * (ppl_local + xpl_local), dim=1, eps=self.eps)
        polarization = F.normalize(
            gate_map * (ppl_local - xpl_local), dim=1, eps=self.eps
        )
        return LocalDescriptorStreams(
            common_map=common,
            polarization_map=polarization,
            common_tokens=self._tokens(common),
            polarization_tokens=self._tokens(polarization),
            fisher_score=fisher,
            polarization_gate=gate,
        )
