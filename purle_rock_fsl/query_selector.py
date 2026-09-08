from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .prototype_bank import DualStreamPrototypeBank


class SelectedQueryDescriptors(NamedTuple):
    common_tokens: torch.Tensor
    polarization_tokens: torch.Tensor
    token_indices: torch.Tensor
    sample_indices: torch.Tensor
    relevance: torch.Tensor
    class_evidence: torch.Tensor


class TaskAwareQuerySelector(nn.Module):
    """Select query tokens using only support-derived class prototypes."""

    def __init__(
        self,
        selected_descriptor_count: int = 24,
        polarization_weight: float = 0.5,
        affinity_weight: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        if selected_descriptor_count < 1:
            raise ValueError("selected_descriptor_count must be positive")
        self.selected_descriptor_count = int(selected_descriptor_count)
        self.polarization_weight = float(polarization_weight)
        self.affinity_weight = float(affinity_weight)
        self.eps = float(eps)

    @staticmethod
    def _query_index(
        query_indices: Sequence[torch.Tensor], sample_count: int
    ) -> torch.Tensor:
        if not query_indices:
            raise ValueError("query_indices must contain at least one tensor")
        for index in query_indices:
            if index.ndim != 1 or index.numel() == 0:
                raise ValueError("each query index tensor must be non-empty and 1D")
            if int(index.min()) < 0 or int(index.max()) >= sample_count:
                raise IndexError("query index is outside the episode batch")
        return torch.cat(query_indices)

    @staticmethod
    def _class_evidence(
        tokens: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        # [query, token, class, prototype]
        similarity = torch.einsum("qtd,ckd->qtck", tokens, prototypes)
        return similarity.max(dim=3).values

    def _relevance(self, class_evidence: torch.Tensor) -> torch.Tensor:
        class_count = class_evidence.size(2)
        if class_count == 1:
            margin = class_evidence[:, :, 0]
            top_affinity = class_evidence[:, :, 0]
        else:
            top2 = class_evidence.topk(k=2, dim=2).values
            margin = top2[:, :, 0] - top2[:, :, 1]
            top_affinity = top2[:, :, 0]
        normalized_affinity = 0.5 * (top_affinity + 1.0)
        return margin + self.affinity_weight * normalized_affinity

    @staticmethod
    def _gather_tokens(tokens: torch.Tensor, index: torch.Tensor):
        return torch.gather(
            tokens,
            1,
            index.unsqueeze(2).expand(-1, -1, tokens.size(2)),
        )

    def forward(
        self,
        common_tokens: torch.Tensor,
        polarization_tokens: torch.Tensor,
        query_indices: Sequence[torch.Tensor],
        prototype_bank: DualStreamPrototypeBank,
    ) -> SelectedQueryDescriptors:
        if common_tokens.shape != polarization_tokens.shape:
            raise ValueError("common and polarization tokens must have equal shapes")
        if common_tokens.ndim != 3:
            raise ValueError("token tensors must have shape [samples, tokens, dim]")
        query_index = self._query_index(query_indices, common_tokens.size(0))
        common = F.normalize(
            common_tokens[query_index], dim=2, eps=self.eps
        )
        polarization = F.normalize(
            polarization_tokens[query_index], dim=2, eps=self.eps
        )
        common_prototypes = F.normalize(
            prototype_bank.common_prototypes, dim=2, eps=self.eps
        )
        polarization_prototypes = F.normalize(
            prototype_bank.polarization_prototypes, dim=2, eps=self.eps
        )

        common_evidence = self._class_evidence(common, common_prototypes)
        polarization_evidence = self._class_evidence(
            polarization, polarization_prototypes
        )
        class_evidence = (
            common_evidence
            + self.polarization_weight * polarization_evidence
        ) / (1.0 + self.polarization_weight)
        relevance = self._relevance(class_evidence)
        if self.training:
            # Training keeps all tokens and lets the downstream relevance
            # softmax assign continuous weights. Evaluation uses fixed top-k.
            selected_index = torch.arange(
                common_tokens.size(1), device=common_tokens.device
            ).unsqueeze(0).expand(common.size(0), -1)
            selected_relevance = relevance
        else:
            selected_count = min(
                self.selected_descriptor_count, common_tokens.size(1)
            )
            selected_relevance, selected_index = relevance.topk(
                k=selected_count, dim=1
            )
        selected_evidence = torch.gather(
            class_evidence,
            1,
            selected_index.unsqueeze(2).expand(
                -1, -1, class_evidence.size(2)
            ),
        )
        return SelectedQueryDescriptors(
            common_tokens=self._gather_tokens(common, selected_index),
            polarization_tokens=self._gather_tokens(
                polarization, selected_index
            ),
            token_indices=selected_index,
            sample_indices=query_index,
            relevance=selected_relevance,
            class_evidence=selected_evidence,
        )
