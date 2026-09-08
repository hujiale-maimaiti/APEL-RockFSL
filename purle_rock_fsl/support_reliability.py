from __future__ import annotations

import math
from typing import List, NamedTuple, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SupportCrossFitFold(NamedTuple):
    support_indices: Tuple[torch.Tensor, ...]
    held_out_indices: torch.Tensor
    targets: torch.Tensor
    fold_index: int


class SupportCrossFitLogits(NamedTuple):
    logits: torch.Tensor
    targets: torch.Tensor
    sample_indices: torch.Tensor
    fold_indices: torch.Tensor


class SupportReliabilitySummary(NamedTuple):
    base_nll: torch.Tensor
    relation_nll: torch.Tensor
    advantage: torch.Tensor
    advantage_confidence: torch.Tensor
    relation_win_rate: torch.Tensor
    class_advantage: torch.Tensor
    class_advantage_confidence: torch.Tensor
    class_relation_win_rate: torch.Tensor
    available: torch.Tensor
    sample_count: torch.Tensor


def balanced_leave_one_shot_out(
    support_indices: Sequence[torch.Tensor],
) -> List[SupportCrossFitFold]:
    """Hold out one support from every class in each balanced fold."""
    if not support_indices:
        raise ValueError("support_indices must contain at least one class")
    shot_count = int(support_indices[0].numel())
    if shot_count <= 1:
        return []
    if any(index.ndim != 1 for index in support_indices):
        raise ValueError("each support index tensor must be 1D")
    if any(int(index.numel()) != shot_count for index in support_indices):
        raise ValueError("balanced cross-fit requires equal shot counts")

    device = support_indices[0].device
    targets = torch.arange(len(support_indices), device=device)
    folds = []
    for fold_index in range(shot_count):
        reduced = []
        held_out = []
        for index in support_indices:
            keep = torch.ones(shot_count, dtype=torch.bool, device=index.device)
            keep[fold_index] = False
            reduced.append(index[keep])
            held_out.append(index[fold_index])
        folds.append(
            SupportCrossFitFold(
                support_indices=tuple(reduced),
                held_out_indices=torch.stack(held_out),
                targets=targets,
                fold_index=fold_index,
            )
        )
    return folds


class SupportReliabilityEstimator(nn.Module):
    """Compare continuous cross-fitted NLL rather than saturated accuracy."""

    def __init__(self, trim_fraction: float = 0.1, eps: float = 1e-6):
        super().__init__()
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        self.trim_fraction = float(trim_fraction)
        self.eps = float(eps)

    @staticmethod
    def unavailable(reference: torch.Tensor) -> SupportReliabilitySummary:
        zero = reference.new_zeros(())
        class_count = reference.size(1) if reference.ndim == 2 else 0
        class_zero = reference.new_zeros((class_count,))
        return SupportReliabilitySummary(
            base_nll=zero,
            relation_nll=zero,
            advantage=zero,
            advantage_confidence=zero,
            relation_win_rate=zero,
            class_advantage=class_zero,
            class_advantage_confidence=class_zero,
            class_relation_win_rate=class_zero,
            available=torch.zeros((), dtype=torch.bool, device=reference.device),
            sample_count=torch.zeros((), dtype=torch.long, device=reference.device),
        )

    def _trimmed_mean(self, values: torch.Tensor) -> torch.Tensor:
        sorted_values = values.sort().values
        trim = int(sorted_values.numel() * self.trim_fraction)
        if trim > 0 and 2 * trim < sorted_values.numel():
            sorted_values = sorted_values[trim:-trim]
        return sorted_values.mean()

    def forward(
        self,
        base_logits: torch.Tensor,
        relation_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> SupportReliabilitySummary:
        if base_logits.shape != relation_logits.shape:
            raise ValueError("base and relation cross-fit logits must match")
        if base_logits.ndim != 2 or targets.shape != base_logits.shape[:1]:
            raise ValueError("invalid cross-fit logits or targets shape")
        if targets.numel() == 0:
            return self.unavailable(base_logits)

        base_nll_rows = F.cross_entropy(
            base_logits, targets, reduction="none"
        )
        relation_nll_rows = F.cross_entropy(
            relation_logits, targets, reduction="none"
        )
        difference = base_nll_rows - relation_nll_rows
        advantage = self._trimmed_mean(difference)
        standard_error = difference.std(unbiased=False) / math.sqrt(
            max(difference.numel(), 1)
        )
        confidence = advantage.abs() / (
            advantage.abs() + standard_error + self.eps
        )
        class_advantages = []
        class_confidences = []
        class_win_rates = []
        for class_index in range(base_logits.size(1)):
            class_difference = difference[targets.eq(class_index)]
            if class_difference.numel() == 0:
                class_advantages.append(difference.new_zeros(()))
                class_confidences.append(difference.new_zeros(()))
                class_win_rates.append(difference.new_zeros(()))
                continue
            class_advantage = self._trimmed_mean(class_difference)
            class_standard_error = class_difference.std(unbiased=False) / math.sqrt(
                class_difference.numel()
            )
            class_confidence = class_advantage.abs() / (
                class_advantage.abs() + class_standard_error + self.eps
            )
            class_advantages.append(class_advantage)
            class_confidences.append(class_confidence)
            class_win_rates.append((class_difference > 0).float().mean())
        return SupportReliabilitySummary(
            base_nll=base_nll_rows.mean().detach(),
            relation_nll=relation_nll_rows.mean().detach(),
            advantage=advantage.detach(),
            advantage_confidence=confidence.detach(),
            relation_win_rate=(difference > 0).float().mean().detach(),
            class_advantage=torch.stack(class_advantages).detach(),
            class_advantage_confidence=torch.stack(class_confidences).detach(),
            class_relation_win_rate=torch.stack(class_win_rates).detach(),
            available=torch.ones((), dtype=torch.bool, device=base_logits.device),
            sample_count=torch.tensor(
                targets.numel(), dtype=torch.long, device=base_logits.device
            ),
        )
