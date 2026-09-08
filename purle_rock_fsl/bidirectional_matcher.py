from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .query_selector import SelectedQueryDescriptors
from .uncertainty_similarity import DualStreamPairwiseSimilarity


class BidirectionalStreamAggregation(NamedTuple):
    score: torch.Tensor
    query_to_prototype: torch.Tensor
    prototype_to_query: torch.Tensor


class DualStreamBidirectionalAggregation(NamedTuple):
    common: BidirectionalStreamAggregation
    polarization: BidirectionalStreamAggregation


class UncertaintyAwareBidirectionalAggregator(nn.Module):
    """Aggregate local pair scores in query-to-support and support-to-query directions."""

    def __init__(
        self,
        query_topk: int = 12,
        reverse_topk: int = 3,
        forward_weight: float = 0.6,
        reliability_log_weight: float = 0.25,
        relevance_temperature: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        if query_topk < 1 or reverse_topk < 1:
            raise ValueError("top-k counts must be positive")
        if not 0.0 <= forward_weight <= 1.0:
            raise ValueError("forward_weight must be in [0, 1]")
        self.query_topk = int(query_topk)
        self.reverse_topk = int(reverse_topk)
        self.forward_weight = float(forward_weight)
        self.reliability_log_weight = float(reliability_log_weight)
        self.relevance_temperature = float(relevance_temperature)
        self.eps = float(eps)

    def _validate(
        self,
        pair_score: torch.Tensor,
        reliability: torch.Tensor,
        relevance: torch.Tensor,
    ):
        if pair_score.ndim != 4:
            raise ValueError("pair_score must have shape [query, token, class, prototype]")
        if reliability.shape != pair_score.shape[2:]:
            raise ValueError("reliability must have shape [class, prototype]")
        if relevance.shape != pair_score.shape[:2]:
            raise ValueError("relevance must have shape [query, token]")

    def _query_to_prototype(
        self, adjusted_score: torch.Tensor, relevance: torch.Tensor
    ) -> torch.Tensor:
        token_score = adjusted_score.max(dim=3).values
        topk = min(self.query_topk, token_score.size(1))
        values, indices = token_score.topk(k=topk, dim=1)
        expanded_relevance = relevance.unsqueeze(2).expand_as(token_score)
        selected_relevance = torch.gather(
            expanded_relevance, dim=1, index=indices
        )
        weights = F.softmax(
            selected_relevance / self.relevance_temperature, dim=1
        )
        return (weights * values).sum(dim=1)

    def _prototype_to_query(
        self,
        adjusted_score: torch.Tensor,
        reliability: torch.Tensor,
        relevance: torch.Tensor,
    ) -> torch.Tensor:
        topk = min(self.reverse_topk, adjusted_score.size(1))
        values, indices = adjusted_score.topk(k=topk, dim=1)
        expanded_relevance = relevance[:, :, None, None].expand_as(
            adjusted_score
        )
        selected_relevance = torch.gather(
            expanded_relevance, dim=1, index=indices
        )
        query_weights = F.softmax(
            selected_relevance / self.relevance_temperature, dim=1
        )
        prototype_match = (query_weights * values).sum(dim=1)
        prototype_weights = reliability / reliability.sum(
            dim=1, keepdim=True
        ).clamp_min(self.eps)
        return (prototype_match * prototype_weights.unsqueeze(0)).sum(dim=2)

    def _aggregate_stream(
        self,
        pair_score: torch.Tensor,
        reliability: torch.Tensor,
        relevance: torch.Tensor,
    ) -> BidirectionalStreamAggregation:
        self._validate(pair_score, reliability, relevance)
        adjusted = pair_score + self.reliability_log_weight * reliability.clamp_min(
            self.eps
        ).log()[None, None, :, :]
        query_to_prototype = self._query_to_prototype(adjusted, relevance)
        prototype_to_query = self._prototype_to_query(
            adjusted, reliability, relevance
        )
        score = (
            self.forward_weight * query_to_prototype
            + (1.0 - self.forward_weight) * prototype_to_query
        )
        return BidirectionalStreamAggregation(
            score=score,
            query_to_prototype=query_to_prototype,
            prototype_to_query=prototype_to_query,
        )

    def forward(
        self,
        pairwise: DualStreamPairwiseSimilarity,
        selected: SelectedQueryDescriptors,
    ) -> DualStreamBidirectionalAggregation:
        common = self._aggregate_stream(
            pairwise.common,
            pairwise.common_reliability,
            selected.relevance,
        )
        polarization = self._aggregate_stream(
            pairwise.polarization,
            pairwise.polarization_reliability,
            selected.relevance,
        )
        return DualStreamBidirectionalAggregation(
            common=common,
            polarization=polarization,
        )
