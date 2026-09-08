from __future__ import annotations

from typing import NamedTuple

import torch

from .baseline_adapter import (
    PURLEFrozenSRCF,
    PURLEPreparedEpisode,
    crossfit_base_support,
    prepare_purle_episode,
)
from .evidence_fusion import EvidenceFusionOutput
from .local_descriptors import LocalDescriptorStreams
from .losses import ComplementaryLossOutput
from .model import PURLEARRCModel
from .relation_expert import PURLERelationExpertOutput
from .support_reliability import SupportReliabilitySummary


class PURLEEpisodeOutput(NamedTuple):
    prepared: PURLEPreparedEpisode
    descriptors: LocalDescriptorStreams
    relation: PURLERelationExpertOutput
    support_reliability: SupportReliabilitySummary
    fusion: EvidenceFusionOutput
    loss: ComplementaryLossOutput | None


def run_purle_episode(
    frozen: PURLEFrozenSRCF,
    model: PURLEARRCModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    shots: int,
    *,
    compute_support_reliability: bool = True,
    compute_loss: bool = True,
) -> PURLEEpisodeOutput:
    prepared = prepare_purle_episode(frozen, images, labels, shots)
    baseline = prepared.baseline
    descriptors = model.encode_descriptors(
        prepared.ppl_features,
        prepared.xpl_features,
        baseline.support_indices,
    )
    relation = model.relation_expert(
        descriptors,
        baseline.support_indices,
        baseline.query_indices,
    )
    if compute_support_reliability and shots > 1:
        base_crossfit = crossfit_base_support(frozen.module, baseline)
        relation_crossfit = model.crossfit_relation_support(
            prepared.ppl_features,
            prepared.xpl_features,
            baseline.support_indices,
        )
        if base_crossfit is None or relation_crossfit is None:
            raise RuntimeError("multi-shot cross-fit unexpectedly unavailable")
        if not torch.equal(base_crossfit.targets, relation_crossfit.targets):
            raise RuntimeError("base/relation cross-fit targets differ")
        if not torch.equal(
            base_crossfit.sample_indices, relation_crossfit.sample_indices
        ):
            raise RuntimeError("base/relation cross-fit sample order differs")
        support_summary = model.support_reliability(
            base_crossfit.logits,
            relation_crossfit.logits,
            base_crossfit.targets,
        )
    else:
        support_summary = model.support_reliability.unavailable(
            baseline.original_logits
        )
    fusion = model.fusion(
        baseline.original_logits,
        relation.relation,
        support_summary,
    )
    loss = (
        model.loss_function(fusion, baseline.query_targets)
        if compute_loss else None
    )
    return PURLEEpisodeOutput(
        prepared=prepared,
        descriptors=descriptors,
        relation=relation,
        support_reliability=support_summary,
        fusion=fusion,
        loss=loss,
    )
