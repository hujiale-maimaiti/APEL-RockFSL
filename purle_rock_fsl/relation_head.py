from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .bidirectional_matcher import DualStreamBidirectionalAggregation
from .query_selector import SelectedQueryDescriptors


class RelationLogitsOutput(NamedTuple):
    logits: torch.Tensor
    common_logits: torch.Tensor
    polarization_logits: torch.Tensor
    polarization_weight: torch.Tensor
    margin: torch.Tensor
    entropy: torch.Tensor
    common_consistency: torch.Tensor
    polarization_consistency: torch.Tensor
    sample_indices: torch.Tensor


class DualStreamRelationHead(nn.Module):
    """Fuse morphology and polarization relation evidence into class logits."""

    def __init__(
        self,
        initial_polarization_weight: float = 0.4,
        initial_temperature: float = 5.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 < initial_polarization_weight < 1.0:
            raise ValueError("initial_polarization_weight must be in (0, 1)")
        self.eps = float(eps)
        initial_logit = math.log(
            initial_polarization_weight / (1.0 - initial_polarization_weight)
        )
        self.polarization_logit = nn.Parameter(torch.tensor(initial_logit))
        self.raw_fisher_scale = nn.Parameter(torch.tensor(0.0))
        self.consistency_scale = nn.Parameter(torch.tensor(1.0))
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(initial_temperature))
        )

    def _standardize(self, score: torch.Tensor) -> torch.Tensor:
        centered = score - score.mean(dim=1, keepdim=True)
        scale = centered.pow(2).mean(dim=1, keepdim=True).sqrt()
        return centered / scale.clamp_min(self.eps)

    def _direction_consistency(self, forward, reverse):
        forward = self._standardize(forward)
        reverse = self._standardize(reverse)
        return torch.exp(-(forward - reverse).abs().mean(dim=1))

    def _polarization_weight(
        self,
        aggregation: DualStreamBidirectionalAggregation,
        polarization_gate: torch.Tensor,
    ):
        common_consistency = self._direction_consistency(
            aggregation.common.query_to_prototype,
            aggregation.common.prototype_to_query,
        )
        polarization_consistency = self._direction_consistency(
            aggregation.polarization.query_to_prototype,
            aggregation.polarization.prototype_to_query,
        )
        fisher_strength = (polarization_gate - 1.0).abs().mean().clamp(0.0, 0.5)
        fisher_strength = 2.0 * fisher_strength
        fisher_scale = F.softplus(self.raw_fisher_scale)
        logit = (
            self.polarization_logit
            + fisher_scale * fisher_strength
            + self.consistency_scale
            * (polarization_consistency - common_consistency)
        )
        return (
            torch.sigmoid(logit),
            common_consistency,
            polarization_consistency,
        )

    @staticmethod
    def _prediction_statistics(logits: torch.Tensor):
        class_count = logits.size(1)
        probability = F.softmax(logits, dim=1)
        if class_count == 1:
            margin = torch.ones_like(probability[:, 0])
            entropy = torch.zeros_like(margin)
        else:
            top2 = probability.topk(k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            entropy = -(
                probability * probability.clamp_min(1e-8).log()
            ).sum(1) / math.log(class_count)
        return margin, entropy

    def forward(
        self,
        aggregation: DualStreamBidirectionalAggregation,
        selected: SelectedQueryDescriptors,
        polarization_gate: torch.Tensor,
    ) -> RelationLogitsOutput:
        if aggregation.common.score.shape != aggregation.polarization.score.shape:
            raise ValueError("common and polarization score tensors must match")
        common = self._standardize(aggregation.common.score)
        polarization = self._standardize(aggregation.polarization.score)
        weight, common_consistency, polarization_consistency = (
            self._polarization_weight(aggregation, polarization_gate)
        )
        fused = (
            (1.0 - weight.unsqueeze(1)) * common
            + weight.unsqueeze(1) * polarization
        )
        temperature = self.log_temperature.exp().clamp(0.5, 20.0)
        logits = temperature * fused
        margin, entropy = self._prediction_statistics(logits)
        return RelationLogitsOutput(
            logits=logits,
            common_logits=temperature * common,
            polarization_logits=temperature * polarization,
            polarization_weight=weight,
            margin=margin,
            entropy=entropy,
            common_consistency=common_consistency,
            polarization_consistency=polarization_consistency,
            sample_indices=selected.sample_indices,
        )
