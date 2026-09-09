from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .evidence_fusion import EvidenceFusionOutput


class ComplementaryLossOutput(NamedTuple):
    loss: torch.Tensor
    fusion_ce: torch.Tensor
    relation_ce: torch.Tensor
    rescue_loss: torch.Tensor
    preservation_loss: torch.Tensor
    base_accuracy: torch.Tensor
    relation_accuracy: torch.Tensor
    fused_accuracy: torch.Tensor
    rescue_rate: torch.Tensor
    damage_rate: torch.Tensor
    mean_gate: torch.Tensor
    gate_supervision_loss: torch.Tensor
    gate_rescue_loss: torch.Tensor
    gate_damage_loss: torch.Tensor
    gate_unproductive_loss: torch.Tensor
    counterfactual_target_mean: torch.Tensor
    rescuable_count: torch.Tensor
    beneficial_count: torch.Tensor
    harmful_count: torch.Tensor


class ComplementaryFusionLoss(nn.Module):
    """Dense classification with explicit rescue and preservation margins."""

    def __init__(
        self,
        relation_weight: float = 0.5,
        rescue_weight: float = 0.75,
        preservation_weight: float = 0.75,
        hard_query_weight: float = 2.0,
        rescue_margin: float = 0.2,
        preservation_tolerance: float = 0.1,
        preservation_confidence: float = 0.7,
        gate_supervision_weight: float = 0.5,
        gate_rescue_weight: float = 1.0,
        gate_damage_weight: float = 1.5,
        counterfactual_grid_size: int = 21,
        gate_unproductive_weight: float = 0.25,
    ):
        super().__init__()
        self.relation_weight = float(relation_weight)
        self.rescue_weight = float(rescue_weight)
        self.preservation_weight = float(preservation_weight)
        self.hard_query_weight = float(hard_query_weight)
        self.rescue_margin = float(rescue_margin)
        self.preservation_tolerance = float(preservation_tolerance)
        self.preservation_confidence = float(preservation_confidence)
        self.gate_supervision_weight = float(gate_supervision_weight)
        self.gate_rescue_weight = float(gate_rescue_weight)
        self.gate_damage_weight = float(gate_damage_weight)
        if counterfactual_grid_size < 2:
            raise ValueError("counterfactual_grid_size must be at least 2")
        self.counterfactual_grid_size = int(counterfactual_grid_size)
        self.gate_unproductive_weight = float(gate_unproductive_weight)

    @staticmethod
    def _true_margin(logits: torch.Tensor, targets: torch.Tensor):
        true_score = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        wrong = logits.clone()
        wrong.scatter_(1, targets.unsqueeze(1), -torch.inf)
        return true_score - wrong.max(dim=1).values

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor):
        if bool(mask.any()):
            return values[mask].mean()
        return values.sum() * 0.0

    def _counterfactual_gate_target(
        self, fusion: EvidenceFusionOutput, targets: torch.Tensor
    ):
        grid = torch.linspace(
            0.0,
            1.0,
            self.counterfactual_grid_size,
            device=fusion.logits.device,
            dtype=fusion.logits.dtype,
        )
        candidate_logits = fusion.base_logits.unsqueeze(0) + (
            grid[:, None, None] * fusion.contrastive_correction.unsqueeze(0)
        )
        candidate_correct = candidate_logits.argmax(2).eq(targets.unsqueeze(0))
        base_wrong = fusion.base_logits.argmax(1).ne(targets)
        allowed = (
            candidate_correct
            & grid[:, None].gt(0.0)
            & base_wrong.unsqueeze(0)
            & fusion.disagreement.bool().unsqueeze(0)
            & fusion.intervention_available.bool().unsqueeze(0)
        )
        rescuable = allowed.any(dim=0)
        first_correct = allowed.float().argmax(dim=0)
        target_gate = grid[first_correct]
        target_gate = torch.where(
            rescuable, target_gate, torch.zeros_like(target_gate)
        )
        return target_gate.detach(), rescuable.detach()

    def forward(
        self,
        fusion: EvidenceFusionOutput,
        targets: torch.Tensor,
    ) -> ComplementaryLossOutput:
        base_logits = fusion.base_logits
        relation_logits = fusion.relation_logits
        fused_logits = fusion.logits
        base_probability = F.softmax(base_logits, dim=1)
        base_true_probability = base_probability.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        hard_weight = 1.0 + self.hard_query_weight * (
            1.0 - base_true_probability
        )
        fusion_rows = F.cross_entropy(
            fused_logits, targets, reduction="none"
        )
        fusion_ce = (hard_weight * fusion_rows).sum() / hard_weight.sum()
        relation_ce = F.cross_entropy(relation_logits, targets)

        base_prediction = base_logits.argmax(1)
        relation_prediction = relation_logits.argmax(1)
        fused_prediction = fused_logits.argmax(1)
        base_correct = base_prediction.eq(targets)
        base_wrong = ~base_correct
        relation_correct = relation_prediction.eq(targets)
        fused_correct = fused_prediction.eq(targets)
        target_gate, rescuable = self._counterfactual_gate_target(
            fusion, targets
        )

        fused_margin = self._true_margin(fused_logits, targets)
        base_margin = self._true_margin(base_logits, targets)
        rescue_loss = self._masked_mean(
            F.relu(self.rescue_margin - fused_margin), rescuable
        )
        preserve_mask = base_correct & (
            base_true_probability >= self.preservation_confidence
        )
        preservation_loss = self._masked_mean(
            F.relu(
                base_margin - self.preservation_tolerance - fused_margin
            ),
            preserve_mask,
        )
        beneficial = base_wrong & relation_correct
        harmful = base_correct & ~relation_correct
        gate = fusion.gate
        gate_rescue_loss = self._masked_mean(
            F.smooth_l1_loss(gate, target_gate, reduction="none"), rescuable
        )
        gate_damage_loss = self._masked_mean(gate.square(), base_correct)
        unproductive = base_wrong & ~rescuable & fusion.disagreement.bool()
        gate_unproductive_loss = self._masked_mean(
            gate.square(), unproductive
        )
        gate_supervision_loss = (
            self.gate_rescue_weight * gate_rescue_loss
            + self.gate_damage_weight * gate_damage_loss
            + self.gate_unproductive_weight * gate_unproductive_loss
        )
        total = (
            fusion_ce
            + self.relation_weight * relation_ce
            + self.rescue_weight * rescue_loss
            + self.preservation_weight * preservation_loss
            + self.gate_supervision_weight * gate_supervision_loss
        )
        rescued = base_wrong & fused_correct
        damaged = base_correct & ~fused_correct
        return ComplementaryLossOutput(
            loss=total,
            fusion_ce=fusion_ce.detach(),
            relation_ce=relation_ce.detach(),
            rescue_loss=rescue_loss.detach(),
            preservation_loss=preservation_loss.detach(),
            base_accuracy=base_correct.float().mean().detach(),
            relation_accuracy=relation_correct.float().mean().detach(),
            fused_accuracy=fused_correct.float().mean().detach(),
            rescue_rate=rescued.float().mean().detach(),
            damage_rate=damaged.float().mean().detach(),
            mean_gate=fusion.gate.mean().detach(),
            gate_supervision_loss=gate_supervision_loss.detach(),
            gate_rescue_loss=gate_rescue_loss.detach(),
            gate_damage_loss=gate_damage_loss.detach(),
            gate_unproductive_loss=gate_unproductive_loss.detach(),
            counterfactual_target_mean=self._masked_mean(
                target_gate, rescuable
            ).detach(),
            rescuable_count=rescuable.sum().detach(),
            beneficial_count=beneficial.sum().detach(),
            harmful_count=harmful.sum().detach(),
        )
