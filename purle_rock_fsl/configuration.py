from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ARRCConfig:
    descriptor_dim: int = 48
    fisher_rank: int = 16
    semantic_scale: float = 0.5
    prototype_count: int = 6
    selected_support_descriptors: int = 24
    selected_query_descriptors: int = 24
    query_topk: int = 12
    reverse_topk: int = 3
    fusion_hidden_dim: int = 8
    relation_loss_weight: float = 0.5
    rescue_loss_weight: float = 0.75
    preservation_loss_weight: float = 0.75
    gate_supervision_weight: float = 0.5
    gate_rescue_weight: float = 1.0
    gate_damage_weight: float = 1.5
    support_prior_cap: float = 0.25
    residual_logit_cap: float = 0.25
    counterfactual_grid_size: int = 21
    gate_unproductive_weight: float = 0.25
