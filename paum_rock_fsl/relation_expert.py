from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import nn

from .bidirectional_matcher import (
    DualStreamBidirectionalAggregation,
    UncertaintyAwareBidirectionalAggregator,
)
from .local_descriptors import LocalDescriptorStreams
from .prototype_bank import (
    DualStreamPrototypeBank,
    TaskAwareMultiPrototypeBuilder,
)
from .query_selector import SelectedQueryDescriptors, TaskAwareQuerySelector
from .relation_head import DualStreamRelationHead, RelationLogitsOutput
from .uncertainty_similarity import (
    DualStreamPairwiseSimilarity,
    UncertaintyAwarePrototypeSimilarity,
)


class PAUMRelationExpertOutput(NamedTuple):
    relation: RelationLogitsOutput
    prototype_bank: DualStreamPrototypeBank
    selected: SelectedQueryDescriptors
    pairwise: DualStreamPairwiseSimilarity
    aggregation: DualStreamBidirectionalAggregation


class PAUMLocalRelationExpert(nn.Module):
    """Complete support-built, query-independent PAUM relation expert."""

    def __init__(
        self,
        prototype_count: int = 6,
        selected_support_descriptors: int = 24,
        selected_query_descriptors: int = 24,
        query_topk: int = 12,
        reverse_topk: int = 3,
    ):
        super().__init__()
        self.prototype_builder = TaskAwareMultiPrototypeBuilder(
            prototype_count=prototype_count,
            selected_descriptor_count=selected_support_descriptors,
        )
        self.query_selector = TaskAwareQuerySelector(
            selected_descriptor_count=selected_query_descriptors
        )
        self.similarity = UncertaintyAwarePrototypeSimilarity()
        self.aggregator = UncertaintyAwareBidirectionalAggregator(
            query_topk=query_topk,
            reverse_topk=reverse_topk,
        )
        self.relation_head = DualStreamRelationHead()

    def build_prototype_bank(
        self,
        descriptors: LocalDescriptorStreams,
        support_indices: Sequence[torch.Tensor],
    ) -> DualStreamPrototypeBank:
        return self.prototype_builder(
            descriptors.common_tokens,
            descriptors.polarization_tokens,
            support_indices,
        )

    def forward_with_bank(
        self,
        descriptors: LocalDescriptorStreams,
        query_indices: Sequence[torch.Tensor],
        prototype_bank: DualStreamPrototypeBank,
    ) -> PAUMRelationExpertOutput:
        selected = self.query_selector(
            descriptors.common_tokens,
            descriptors.polarization_tokens,
            query_indices,
            prototype_bank,
        )
        pairwise = self.similarity(selected, prototype_bank)
        aggregation = self.aggregator(pairwise, selected)
        relation = self.relation_head(
            aggregation,
            selected,
            descriptors.polarization_gate,
        )
        return PAUMRelationExpertOutput(
            relation=relation,
            prototype_bank=prototype_bank,
            selected=selected,
            pairwise=pairwise,
            aggregation=aggregation,
        )

    def forward(
        self,
        descriptors: LocalDescriptorStreams,
        support_indices: Sequence[torch.Tensor],
        query_indices: Sequence[torch.Tensor],
    ) -> PAUMRelationExpertOutput:
        prototype_bank = self.build_prototype_bank(
            descriptors, support_indices
        )
        return self.forward_with_bank(
            descriptors, query_indices, prototype_bank
        )
