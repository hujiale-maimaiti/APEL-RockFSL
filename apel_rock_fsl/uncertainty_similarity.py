from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .prototype_bank import DualStreamPrototypeBank
from .query_selector import SelectedQueryDescriptors


class DualStreamPairwiseSimilarity(NamedTuple):
    common: torch.Tensor
    polarization: torch.Tensor
    common_reliability: torch.Tensor
    polarization_reliability: torch.Tensor


class UncertaintyAwarePrototypeSimilarity(nn.Module):
    """Compute diagonal-uncertainty scores for every query-token/prototype pair."""

    def __init__(
        self,
        query_noise: float = 0.05,
        count_prior: float = 2.0,
        min_pair_variance: float = 1e-4,
        max_pair_variance: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if query_noise <= 0:
            raise ValueError("query_noise must be positive")
        self.query_noise = float(query_noise)
        self.count_prior = float(count_prior)
        self.min_pair_variance = float(min_pair_variance)
        self.max_pair_variance = float(max_pair_variance)
        self.eps = float(eps)
        # One scalar per stream lets training calibrate support dispersion
        # without learning a high-capacity distance function.
        self.log_variance_scale = nn.Parameter(torch.zeros(2))

    def _stream_similarity(
        self,
        query_tokens: torch.Tensor,
        prototypes: torch.Tensor,
        prototype_variance: torch.Tensor,
        effective_count: torch.Tensor,
        stream_index: int,
    ):
        if prototypes.shape != prototype_variance.shape:
            raise ValueError("prototype and variance tensors must have equal shapes")
        if effective_count.shape != prototypes.shape[:2]:
            raise ValueError("effective_count must have shape [class, prototype]")
        if query_tokens.size(2) != prototypes.size(2):
            raise ValueError("query and prototype descriptor dimensions differ")

        query = F.normalize(query_tokens, dim=2, eps=self.eps)
        prototype = F.normalize(prototypes, dim=2, eps=self.eps)
        variance_scale = self.log_variance_scale[stream_index].exp()
        support_variance = (prototype_variance * variance_scale).clamp(
            min=self.min_pair_variance,
            max=self.max_pair_variance,
        )
        pair_variance = (support_variance + self.query_noise).clamp(
            min=self.min_pair_variance,
            max=self.max_pair_variance,
        )

        squared_error = (
            query[:, :, None, None, :] - prototype[None, None, :, :, :]
        ).pow(2)
        negative_log_likelihood = 0.5 * (
            squared_error / pair_variance[None, None, :, :, :]
            + pair_variance.log()[None, None, :, :, :]
        ).mean(dim=4)
        similarity = -negative_log_likelihood

        count_reliability = effective_count / (
            effective_count + self.count_prior
        )
        uncertainty_reliability = torch.exp(
            -torch.log1p(support_variance / self.query_noise).mean(dim=2)
        )
        reliability = (count_reliability * uncertainty_reliability).clamp(
            min=self.eps, max=1.0
        )
        return similarity, reliability

    def forward(
        self,
        selected: SelectedQueryDescriptors,
        prototype_bank: DualStreamPrototypeBank,
    ) -> DualStreamPairwiseSimilarity:
        common, common_reliability = self._stream_similarity(
            selected.common_tokens,
            prototype_bank.common_prototypes,
            prototype_bank.common_variance,
            prototype_bank.effective_count,
            stream_index=0,
        )
        polarization, polarization_reliability = self._stream_similarity(
            selected.polarization_tokens,
            prototype_bank.polarization_prototypes,
            prototype_bank.polarization_variance,
            prototype_bank.effective_count,
            stream_index=1,
        )
        return DualStreamPairwiseSimilarity(
            common=common,
            polarization=polarization,
            common_reliability=common_reliability,
            polarization_reliability=polarization_reliability,
        )
