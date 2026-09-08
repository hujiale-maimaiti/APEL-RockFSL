from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .relation_head import RelationLogitsOutput
from .support_reliability import SupportReliabilitySummary


class EvidenceFusionOutput(NamedTuple):
    logits: torch.Tensor
    base_logits: torch.Tensor
    relation_logits: torch.Tensor
    contrastive_correction: torch.Tensor
    gate: torch.Tensor
    correction_scale: torch.Tensor
    support_signal: torch.Tensor
    gate_features: torch.Tensor
    confidence_advantage: torch.Tensor
    disagreement: torch.Tensor
    intervention_available: torch.Tensor
    monotonic_score: torch.Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class AsymmetricRescueRiskFusion(nn.Module):
    """Use relation evidence only when its confidence exceeds the frozen base."""

    def __init__(
        self,
        hidden_dim: int = 8,
        initial_gate: float = 0.05,
        support_temperature: float = 0.25,
        support_prior_cap: float = 0.25,
        residual_logit_cap: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be in (0, 1)")
        self.support_temperature = float(support_temperature)
        if not 0.0 <= support_prior_cap <= 1.0:
            raise ValueError("support_prior_cap must be in [0, 1]")
        self.support_prior_cap = float(support_prior_cap)
        if residual_logit_cap < 0.0:
            raise ValueError("residual_logit_cap must be non-negative")
        self.residual_logit_cap = float(residual_logit_cap)
        self.eps = float(eps)
        self.initial_gate_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.raw_support_scale = nn.Parameter(torch.tensor(-1.3862944))
        self.raw_relative_weights = nn.Parameter(
            torch.full((2,), _inverse_softplus(0.5))
        )
        self.query_gate = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.trunc_normal_(self.query_gate[0].weight, std=0.02)
        nn.init.zeros_(self.query_gate[0].bias)
        nn.init.zeros_(self.query_gate[-1].weight)
        nn.init.zeros_(self.query_gate[-1].bias)

    def _standardize(self, logits: torch.Tensor):
        mean = logits.mean(dim=1, keepdim=True)
        centered = logits - mean
        scale = centered.pow(2).mean(dim=1, keepdim=True).sqrt()
        return centered / scale.clamp_min(self.eps), mean, scale

    @staticmethod
    def _confidence_statistics(logits: torch.Tensor):
        probability = F.softmax(logits, dim=1)
        class_count = logits.size(1)
        if class_count == 1:
            zero = torch.zeros_like(probability[:, 0])
            one = torch.ones_like(probability[:, 0])
            return zero, one, one
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(1) / math.log(class_count)
        top2 = probability.topk(k=2, dim=1).values
        return entropy, top2[:, 0] - top2[:, 1], top2[:, 0]

    def _gate_features(
        self,
        base_logits: torch.Tensor,
        relation: RelationLogitsOutput,
        support: SupportReliabilitySummary,
    ):
        base_entropy, base_margin, _ = self._confidence_statistics(
            base_logits
        )
        relation_entropy = relation.entropy
        relation_margin = relation.margin
        disagreement = (
            base_logits.argmax(1) != relation.logits.argmax(1)
        ).float()
        base_prediction = base_logits.argmax(1)
        relation_prediction = relation.logits.argmax(1)
        query_count = base_logits.size(0)
        available = support.available.float().expand(query_count)
        support_confidence = support.advantage_confidence.expand(query_count)
        support_win = (2.0 * support.relation_win_rate - 1.0).expand(query_count)
        relation_class_advantage = support.class_advantage.gather(
            0, relation_prediction
        ) * available
        relation_class_confidence = support.class_advantage_confidence.gather(
            0, relation_prediction
        ) * available
        relation_class_win = (
            2.0 * support.class_relation_win_rate.gather(0, relation_prediction) - 1.0
        ) * available
        base_class_advantage = support.class_advantage.gather(
            0, base_prediction
        ) * available
        mean_consistency = 0.5 * (
            relation.common_consistency + relation.polarization_consistency
        )
        consistency_gap = (
            relation.polarization_consistency - relation.common_consistency
        )
        margin_advantage = relation_margin - base_margin
        entropy_advantage = base_entropy - relation_entropy
        features = torch.stack(
            [
                disagreement,
                mean_consistency,
                consistency_gap,
                support_confidence,
                support_win,
                available,
                torch.tanh(relation_class_advantage / self.support_temperature),
                relation_class_confidence,
                relation_class_win,
                torch.tanh(
                    (relation_class_advantage - base_class_advantage)
                    / self.support_temperature
                ),
            ],
            dim=1,
        )
        relative = torch.stack(
            [margin_advantage, entropy_advantage], dim=1
        )
        class_signal = (
            torch.tanh(relation_class_advantage / self.support_temperature)
            * relation_class_confidence
        )
        return features, relative, disagreement, class_signal

    def forward(
        self,
        base_logits: torch.Tensor,
        relation: RelationLogitsOutput,
        support: SupportReliabilitySummary,
    ) -> EvidenceFusionOutput:
        if base_logits.shape != relation.logits.shape:
            raise ValueError("base and relation logits must have equal shapes")
        frozen_base = base_logits.detach()
        base_standard, _, base_scale = self._standardize(frozen_base)
        relation_standard, _, _ = self._standardize(relation.logits)
        contrastive_correction = base_scale.clamp_min(self.eps) * (
            relation_standard - base_standard
        )

        global_support_signal = (
            torch.tanh(support.advantage / self.support_temperature)
            * support.advantage_confidence
            * support.available.float()
        )
        gate_features, relative_features, disagreement, class_signal = self._gate_features(
            frozen_base, relation, support
        )
        gate_features = gate_features.detach()
        relative_features = relative_features.detach()
        disagreement = disagreement.detach()
        support_signal = 0.5 * (
            global_support_signal.expand_as(class_signal) + class_signal
        ).detach()
        support_scale = self.support_prior_cap * torch.sigmoid(
            self.raw_support_scale
        )
        relative_weights = F.softplus(self.raw_relative_weights)
        monotonic_score = (
            relative_features * relative_weights.unsqueeze(0)
        ).sum(dim=1)
        residual = self.residual_logit_cap * torch.tanh(
            self.query_gate(gate_features).squeeze(1)
        )
        gate_logit = (
            self.initial_gate_logit
            + support_scale * support_signal
            + monotonic_score
            + residual
        )
        gate = (
            torch.sigmoid(gate_logit)
            * support.available.float()
            * disagreement
        )
        correction_scale = gate.new_ones(())
        fused_logits = frozen_base + gate.unsqueeze(1) * contrastive_correction
        return EvidenceFusionOutput(
            logits=fused_logits,
            base_logits=frozen_base,
            relation_logits=relation.logits,
            contrastive_correction=contrastive_correction,
            gate=gate,
            correction_scale=correction_scale,
            support_signal=support_signal,
            gate_features=gate_features,
            confidence_advantage=relative_features.mean(dim=1),
            disagreement=disagreement,
            intervention_available=support.available.float().expand_as(gate).detach(),
            monotonic_score=monotonic_score,
        )
