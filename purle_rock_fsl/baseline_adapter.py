from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .base_fusion_model import ShotAwareReliabilityConstrainedFusion
from .multiscale_backbone import (
    FGKMultiScaleResNet18Backbone,
    MultiScaleFeatureMaps,
)
from .resnet_backbone import FGKMiniResNet18Backbone
from .support_reliability import (
    SupportCrossFitLogits,
    balanced_leave_one_shot_out,
)
from .utils import load_checkpoint


@dataclass(frozen=True)
class BaselineEpisode:
    """Frozen base-model evidence exposed to the PURLE relation expert."""

    classes: torch.Tensor
    support_indices: tuple[torch.Tensor, ...]
    query_indices: tuple[torch.Tensor, ...]
    flat_query_indices: torch.Tensor
    query_targets: torch.Tensor
    ppl_map: torch.Tensor
    xpl_map: torch.Tensor
    tdpf_embedding: torch.Tensor
    tdpf_ppl_embedding: torch.Tensor
    tdpf_xpl_embedding: torch.Tensor
    cupm_embedding: torch.Tensor
    cupm_variance: torch.Tensor
    ppl_embedding: torch.Tensor
    xpl_embedding: torch.Tensor
    fisher_score: torch.Tensor
    channel_gate: torch.Tensor
    tdpf_logits: torch.Tensor
    cupm_logits: torch.Tensor
    original_logits: torch.Tensor
    original_alpha: torch.Tensor
    router_features: torch.Tensor
    evidence_strength: torch.Tensor

    @property
    def ways(self) -> int:
        return len(self.support_indices)

    @property
    def shots(self) -> int:
        counts = {int(index.numel()) for index in self.support_indices}
        if len(counts) != 1:
            raise ValueError("all classes must have the same support count")
        return next(iter(counts))

    def validate(self) -> "BaselineEpisode":
        if self.ways < 2 or self.shots <= 0:
            raise ValueError("an episode needs at least two classes")
        if self.ppl_map.ndim != 4 or self.ppl_map.shape != self.xpl_map.shape:
            raise ValueError("PPL/XPL maps must share shape [images, C, H, W]")
        image_count, channels = self.ppl_map.shape[:2]
        embedding_shape = self.tdpf_embedding.shape
        if len(embedding_shape) != 2 or embedding_shape[0] != image_count:
            raise ValueError("tdpf_embedding has an invalid shape")
        aligned = (
            self.tdpf_ppl_embedding,
            self.tdpf_xpl_embedding,
            self.cupm_embedding,
            self.cupm_variance,
            self.ppl_embedding,
            self.xpl_embedding,
        )
        if any(value.shape != embedding_shape for value in aligned):
            raise ValueError("base-model embeddings and variance must align")
        if tuple(self.fisher_score.shape) != (channels,):
            raise ValueError("fisher_score must have one value per channel")
        if tuple(self.channel_gate.shape) != (channels,):
            raise ValueError("channel_gate must have one value per channel")
        query_count = int(self.flat_query_indices.numel())
        expected_logits = (query_count, self.ways)
        logits = (self.tdpf_logits, self.cupm_logits, self.original_logits)
        if any(tuple(value.shape) != expected_logits for value in logits):
            raise ValueError("all branch logits must be [queries, ways]")
        if tuple(self.original_alpha.shape) != (query_count,):
            raise ValueError("original_alpha must have one value per query")
        if tuple(self.query_targets.shape) != (query_count,):
            raise ValueError("query_targets must have one value per query")
        if self.router_features.shape[0] != query_count:
            raise ValueError("router_features must have one row per query")
        if tuple(self.evidence_strength.shape) != (query_count,):
            raise ValueError("evidence_strength must have one value per query")
        index_groups = (*self.support_indices, *self.query_indices)
        if any(
            index.ndim != 1
            or index.numel() == 0
            or int(index.min()) < 0
            or int(index.max()) >= image_count
            for index in index_groups
        ):
            raise ValueError("episode indices are invalid")
        tensors = (
            self.ppl_map,
            self.xpl_map,
            self.tdpf_embedding,
            *aligned,
            self.fisher_score,
            self.channel_gate,
            *logits,
            self.original_alpha,
            self.router_features,
            self.evidence_strength,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("baseline state must contain finite tensors")
        if bool((self.cupm_variance <= 0.0).any()):
            raise ValueError("CUPM variance must be positive")
        return self


@dataclass(frozen=True)
class FrozenBaseModel:
    backbone: FGKMiniResNet18Backbone
    module: ShotAwareReliabilityConstrainedFusion
    checkpoint: dict[str, Any]


def _build_base_fusion(config: dict[str, Any]) -> ShotAwareReliabilityConstrainedFusion:
    return ShotAwareReliabilityConstrainedFusion(
        in_channels=512,
        embedding_dim=128,
        gate_rank=16,
        interaction_rank=16,
        uncertainty_rank=32,
        branch_loss_weight=float(config.get("branch_loss_weight", 0.5)),
        router_loss_weight=float(config.get("router_loss_weight", 0.10)),
        prior_loss_weight=float(config.get("prior_loss_weight", 0.02)),
        regret_loss_weight=float(config.get("regret_loss_weight", 0.15)),
        correction_radius_1shot=float(config.get("correction_radius_1shot", 0.12)),
        correction_radius_multishot=float(
            config.get("correction_radius_multishot", 0.10)
        ),
    )


def load_frozen_base_model(
    checkpoint_path: str | Path, *, device: torch.device
) -> FrozenBaseModel:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    if checkpoint.get("model") != "SRCF_TDPF_CUPM":
        raise RuntimeError(
            "PURLE requires SRCF_TDPF_CUPM, got {!r}".format(
                checkpoint.get("model")
            )
        )
    if "backbone" not in checkpoint or "srcf_module" not in checkpoint:
        raise KeyError("checkpoint must contain backbone and srcf_module")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise TypeError("checkpoint config must be a dictionary")
    backbone = FGKMiniResNet18Backbone().to(device)
    module = _build_base_fusion(config).to(device)
    backbone.load_state_dict(checkpoint["backbone"], strict=True)
    module.load_state_dict(checkpoint["srcf_module"], strict=True)
    backbone.eval()
    module.eval()
    for parameter in (*backbone.parameters(), *module.parameters()):
        parameter.requires_grad_(False)
    return FrozenBaseModel(
        backbone=backbone,
        module=module,
        checkpoint=checkpoint,
    )


@torch.no_grad()
def extract_baseline_episode(
    module: ShotAwareReliabilityConstrainedFusion,
    ppl_map: torch.Tensor,
    xpl_map: torch.Tensor,
    target: torch.Tensor,
    n_support: int,
) -> BaselineEpisode:
    classes, support_indices, query_indices, query_targets = module._episode_indices(
        target, n_support
    )
    query_targets = torch.cat(query_targets)
    flat_query_indices = torch.cat(query_indices)
    encoded = module._encode_experts(ppl_map, xpl_map, support_indices)
    tdpf_logits = module._euclidean_logits(
        encoded["tdpf"], support_indices, query_indices
    )
    cupm_logits = module._uncertainty_logits(
        encoded["cupm"], encoded["variance"], support_indices, query_indices
    )
    tdpf_probability = F.softmax(tdpf_logits, dim=1)
    cupm_probability = F.softmax(cupm_logits, dim=1)
    router_features = module._router_features(
        tdpf_logits,
        cupm_logits,
        encoded,
        support_indices,
        query_indices,
        n_support,
    )
    prior = router_features.new_full(
        (router_features.size(0),), module._shot_prior(n_support)
    )
    radius = module._correction_radius(n_support)
    evidence_strength = module._evidence_strength(router_features.detach())
    router_residual = torch.tanh(module.router(router_features.detach()).squeeze(1))
    original_alpha = (prior + radius * evidence_strength * router_residual).clamp(
        module._shot_prior(n_support) - radius,
        module._shot_prior(n_support) + radius,
    )
    original_logits = (
        (1.0 - original_alpha.unsqueeze(1)) * tdpf_probability
        + original_alpha.unsqueeze(1) * cupm_probability
    ).clamp_min(module.eps).log()
    tdpf_ppl_embedding = module.tdpf_projection(module.pool(ppl_map).flatten(1))
    tdpf_xpl_embedding = module.tdpf_projection(module.pool(xpl_map).flatten(1))
    fisher_score = module._task_fisher_score(ppl_map, xpl_map, support_indices)
    channel_gate = torch.sigmoid(module.task_gate(fisher_score)).flatten()
    return BaselineEpisode(
        classes=classes,
        support_indices=tuple(support_indices),
        query_indices=tuple(query_indices),
        flat_query_indices=flat_query_indices,
        query_targets=query_targets,
        ppl_map=ppl_map,
        xpl_map=xpl_map,
        tdpf_embedding=encoded["tdpf"],
        tdpf_ppl_embedding=tdpf_ppl_embedding,
        tdpf_xpl_embedding=tdpf_xpl_embedding,
        cupm_embedding=encoded["cupm"],
        cupm_variance=encoded["variance"],
        ppl_embedding=encoded["ppl"],
        xpl_embedding=encoded["xpl"],
        fisher_score=fisher_score,
        channel_gate=channel_gate,
        tdpf_logits=tdpf_logits,
        cupm_logits=cupm_logits,
        original_logits=original_logits,
        original_alpha=original_alpha,
        router_features=router_features,
        evidence_strength=evidence_strength,
    ).validate()


@dataclass(frozen=True)
class PURLEFrozenSRCF:
    backbone: FGKMultiScaleResNet18Backbone
    module: ShotAwareReliabilityConstrainedFusion
    checkpoint: dict[str, Any]


@dataclass(frozen=True)
class PURLEPreparedEpisode:
    baseline: BaselineEpisode
    ppl_features: MultiScaleFeatureMaps
    xpl_features: MultiScaleFeatureMaps


def load_purle_frozen_srcf(
    checkpoint_path: str | Path, *, device: torch.device
) -> PURLEFrozenSRCF:
    source = load_frozen_base_model(checkpoint_path, device=device)
    backbone = FGKMultiScaleResNet18Backbone().to(device)
    backbone.load_state_dict(source.backbone.state_dict(), strict=True)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return PURLEFrozenSRCF(
        backbone=backbone,
        module=source.module,
        checkpoint=source.checkpoint,
    )


@torch.no_grad()
def prepare_purle_episode(
    frozen: PURLEFrozenSRCF,
    images: torch.Tensor,
    labels: torch.Tensor,
    shots: int,
) -> PURLEPreparedEpisode:
    ppl_features = frozen.backbone.forward_multiscale(images[:, 3:6])
    xpl_features = frozen.backbone.forward_multiscale(images[:, 0:3])
    baseline = extract_baseline_episode(
        frozen.module,
        ppl_features.layer4,
        xpl_features.layer4,
        labels,
        shots,
    )
    return PURLEPreparedEpisode(
        baseline=baseline,
        ppl_features=ppl_features,
        xpl_features=xpl_features,
    )


@torch.no_grad()
def fixed_fusion_logits_for_indices(
    module: ShotAwareReliabilityConstrainedFusion,
    state: BaselineEpisode,
    support_indices: Sequence[torch.Tensor],
    query_indices: Sequence[torch.Tensor],
) -> torch.Tensor:
    # TDPF's Fisher gate is support-conditioned. Recompute every fold so the
    # held-out sample cannot influence the representation used to predict it.
    encoded = module._encode_experts(
        state.ppl_map,
        state.xpl_map,
        support_indices,
    )
    tdpf_logits = module._euclidean_logits(
        encoded["tdpf"], support_indices, query_indices
    )
    cupm_logits = module._uncertainty_logits(
        encoded["cupm"], encoded["variance"], support_indices, query_indices
    )
    features = module._router_features(
        tdpf_logits,
        cupm_logits,
        encoded,
        support_indices,
        query_indices,
        int(support_indices[0].numel()),
    )
    prior = features.new_full(
        (features.size(0),), module._shot_prior(int(support_indices[0].numel()))
    )
    radius = module._correction_radius(int(support_indices[0].numel()))
    strength = module._evidence_strength(features)
    residual = torch.tanh(module.router(features).squeeze(1))
    alpha = (prior + radius * strength * residual).clamp(
        module._shot_prior(int(support_indices[0].numel())) - radius,
        module._shot_prior(int(support_indices[0].numel())) + radius,
    )
    probability = (
        (1.0 - alpha.unsqueeze(1)) * F.softmax(tdpf_logits, dim=1)
        + alpha.unsqueeze(1) * F.softmax(cupm_logits, dim=1)
    ).clamp_min(module.eps)
    return probability.log()


@torch.no_grad()
def crossfit_base_support(
    module: ShotAwareReliabilityConstrainedFusion,
    state: BaselineEpisode,
) -> SupportCrossFitLogits | None:
    folds = balanced_leave_one_shot_out(state.support_indices)
    if not folds:
        return None
    logits = []
    targets = []
    samples = []
    fold_ids = []
    for fold in folds:
        logits.append(
            fixed_fusion_logits_for_indices(
                module,
                state,
                fold.support_indices,
                [fold.held_out_indices],
            )
        )
        targets.append(fold.targets)
        samples.append(fold.held_out_indices)
        fold_ids.append(torch.full_like(fold.targets, fold.fold_index))
    return SupportCrossFitLogits(
        logits=torch.cat(logits),
        targets=torch.cat(targets),
        sample_indices=torch.cat(samples),
        fold_indices=torch.cat(fold_ids),
    )
