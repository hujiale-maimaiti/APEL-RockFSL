"""PURLE-ARRC few-shot rock thin-section classification package."""

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
    PURLELocalRelationExpert,
    PURLERelationExpertOutput,
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
from .model import PURLEARRCModel
from .baseline_adapter import (
    PURLEFrozenSRCF,
    PURLEPreparedEpisode,
    crossfit_base_support,
    fixed_fusion_logits_for_indices,
    load_purle_frozen_srcf,
    prepare_purle_episode,
)
from .runtime import PURLEEpisodeOutput, run_purle_episode
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
    "PURLELocalRelationExpert",
    "PURLEARRCModel",
    "ARRCConfig",
    "PURLEFrozenSRCF",
    "PURLEPreparedEpisode",
    "PURLEEpisodeOutput",
    "PURLERelationExpertOutput",
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
    "load_purle_frozen_srcf",
    "prepare_purle_episode",
    "run_purle_episode",
]
