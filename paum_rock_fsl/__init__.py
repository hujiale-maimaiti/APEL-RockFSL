"""PAUM few-shot rock thin-section classification package."""

from .paired_dataset import NJURockSynchronizedPairDataset
from .paired_transforms import SynchronizedPairTransform
from .multiscale_backbone import (
    FGKMultiScaleResNet18Backbone,
    MultiScaleFeatureMaps,
)
from .local_descriptors import (
    DualStreamLocalDescriptor,
    LocalDescriptorStreams,
)
from .prototype_bank import (
    DualStreamPrototypeBank,
    TaskAwareMultiPrototypeBuilder,
)
from .query_selector import (
    SelectedQueryDescriptors,
    TaskAwareQuerySelector,
)
from .uncertainty_similarity import (
    DualStreamPairwiseSimilarity,
    UncertaintyAwarePrototypeSimilarity,
)
from .bidirectional_matcher import (
    BidirectionalStreamAggregation,
    DualStreamBidirectionalAggregation,
    UncertaintyAwareBidirectionalAggregator,
)
from .relation_head import (
    DualStreamRelationHead,
    RelationLogitsOutput,
)
from .relation_expert import (
    PAUMLocalRelationExpert,
    PAUMRelationExpertOutput,
)
from .support_reliability import (
    SupportCrossFitFold,
    SupportCrossFitLogits,
    SupportReliabilityEstimator,
    SupportReliabilitySummary,
    balanced_leave_one_shot_out,
)
from .evidence_fusion import (
    AsymmetricRescueRiskFusion,
    EvidenceFusionOutput,
)
from .configuration import ARRCConfig
from .model import PAUMARRCModel
from .baseline_adapter import (
    PAUMFrozenSRCF,
    PAUMPreparedEpisode,
    crossfit_base_support,
    fixed_fusion_logits_for_indices,
    load_paum_frozen_srcf,
    prepare_paum_episode,
)
from .runtime import PAUMEpisodeOutput, run_paum_episode
from .losses import (
    ComplementaryFusionLoss,
    ComplementaryLossOutput,
)

__all__ = [
    "FGKMultiScaleResNet18Backbone",
    "DualStreamLocalDescriptor",
    "DualStreamPrototypeBank",
    "DualStreamPairwiseSimilarity",
    "DualStreamRelationHead",
    "AsymmetricRescueRiskFusion",
    "ComplementaryFusionLoss",
    "ComplementaryLossOutput",
    "BidirectionalStreamAggregation",
    "DualStreamBidirectionalAggregation",
    "LocalDescriptorStreams",
    "MultiScaleFeatureMaps",
    "NJURockSynchronizedPairDataset",
    "PAUMLocalRelationExpert",
    "PAUMARRCModel",
    "ARRCConfig",
    "PAUMFrozenSRCF",
    "PAUMPreparedEpisode",
    "PAUMEpisodeOutput",
    "PAUMRelationExpertOutput",
    "EvidenceFusionOutput",
    "SelectedQueryDescriptors",
    "SupportCrossFitFold",
    "SupportCrossFitLogits",
    "SupportReliabilityEstimator",
    "SupportReliabilitySummary",
    "RelationLogitsOutput",
    "SynchronizedPairTransform",
    "TaskAwareMultiPrototypeBuilder",
    "TaskAwareQuerySelector",
    "UncertaintyAwareBidirectionalAggregator",
    "UncertaintyAwarePrototypeSimilarity",
    "balanced_leave_one_shot_out",
    "crossfit_base_support",
    "fixed_fusion_logits_for_indices",
    "load_paum_frozen_srcf",
    "prepare_paum_episode",
    "run_paum_episode",
]
