from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .configuration import ARRCConfig
from .evidence_fusion import AsymmetricRescueRiskFusion
from .local_descriptors import DualStreamLocalDescriptor
from .losses import ComplementaryFusionLoss
from .multiscale_backbone import MultiScaleFeatureMaps
from .relation_expert import PURLELocalRelationExpert
from .support_reliability import SupportReliabilityEstimator
from .support_reliability import (
    SupportCrossFitLogits,
    balanced_leave_one_shot_out,
)


class PURLEARRCModel(nn.Module):
    """PURLE relation expert with an asymmetric rescue-risk controller."""

    def __init__(self, config: ARRCConfig):
        super().__init__()
        self.config = config
        self.descriptor = DualStreamLocalDescriptor(
            descriptor_dim=config.descriptor_dim,
            fisher_rank=config.fisher_rank,
            semantic_scale=config.semantic_scale,
        )
        self.relation_expert = PURLELocalRelationExpert(
            prototype_count=config.prototype_count,
            selected_support_descriptors=config.selected_support_descriptors,
            selected_query_descriptors=config.selected_query_descriptors,
            query_topk=config.query_topk,
            reverse_topk=config.reverse_topk,
        )
        self.support_reliability = SupportReliabilityEstimator()
        self.fusion = AsymmetricRescueRiskFusion(
            hidden_dim=config.fusion_hidden_dim,
            support_prior_cap=config.support_prior_cap,
            residual_logit_cap=config.residual_logit_cap,
        )
        self.loss_function = ComplementaryFusionLoss(
            relation_weight=config.relation_loss_weight,
            rescue_weight=config.rescue_loss_weight,
            preservation_weight=config.preservation_loss_weight,
            gate_supervision_weight=config.gate_supervision_weight,
            gate_rescue_weight=config.gate_rescue_weight,
            gate_damage_weight=config.gate_damage_weight,
            counterfactual_grid_size=config.counterfactual_grid_size,
            gate_unproductive_weight=config.gate_unproductive_weight,
        )

    def encode_descriptors(
        self,
        ppl_features: MultiScaleFeatureMaps,
        xpl_features: MultiScaleFeatureMaps,
        support_indices: Sequence[torch.Tensor],
    ):
        return self.descriptor(
            ppl_features, xpl_features, support_indices
        )

    @torch.no_grad()
    def crossfit_relation_support(
        self,
        ppl_features: MultiScaleFeatureMaps,
        xpl_features: MultiScaleFeatureMaps,
        support_indices: Sequence[torch.Tensor],
    ) -> SupportCrossFitLogits | None:
        folds = balanced_leave_one_shot_out(support_indices)
        if not folds:
            return None
        relation_was_training = self.relation_expert.training
        self.relation_expert.eval()
        logits = []
        targets = []
        sample_indices = []
        fold_indices = []
        try:
            for fold in folds:
                descriptors = self.encode_descriptors(
                    ppl_features,
                    xpl_features,
                    fold.support_indices,
                )
                output = self.relation_expert(
                    descriptors,
                    fold.support_indices,
                    [fold.held_out_indices],
                )
                logits.append(output.relation.logits)
                targets.append(fold.targets)
                sample_indices.append(fold.held_out_indices)
                fold_indices.append(
                    torch.full_like(fold.targets, fold.fold_index)
                )
        finally:
            self.relation_expert.train(relation_was_training)
        return SupportCrossFitLogits(
            logits=torch.cat(logits),
            targets=torch.cat(targets),
            sample_indices=torch.cat(sample_indices),
            fold_indices=torch.cat(fold_indices),
        )

    def parameter_report(self):
        groups = {
            "local_descriptor": self.descriptor,
            "relation_expert": self.relation_expert,
            "evidence_fusion": self.fusion,
        }
        report = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        report["total"] = sum(parameter.numel() for parameter in self.parameters())
        report["trainable"] = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return report
