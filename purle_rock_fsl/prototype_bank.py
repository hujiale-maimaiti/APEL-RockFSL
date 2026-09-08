from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DualStreamPrototypeBank(NamedTuple):
    common_prototypes: torch.Tensor
    polarization_prototypes: torch.Tensor
    common_variance: torch.Tensor
    polarization_variance: torch.Tensor
    effective_count: torch.Tensor
    selected_scores: torch.Tensor
    selected_sample_indices: torch.Tensor


class TaskAwareMultiPrototypeBuilder(nn.Module):
    """Select discriminative support tokens and build class-wise local banks."""

    def __init__(
        self,
        prototype_count: int = 6,
        selected_descriptor_count: int = 24,
        polarization_selection_weight: float = 0.5,
        assignment_temperature: float = 0.15,
        clustering_iterations: int = 2,
        variance_prior_strength: float = 4.0,
        min_variance: float = 1e-4,
        eps: float = 1e-6,
    ):
        super().__init__()
        if prototype_count < 1:
            raise ValueError("prototype_count must be positive")
        if selected_descriptor_count < prototype_count:
            raise ValueError(
                "selected_descriptor_count must be at least prototype_count"
            )
        self.prototype_count = int(prototype_count)
        self.selected_descriptor_count = int(selected_descriptor_count)
        self.polarization_selection_weight = float(
            polarization_selection_weight
        )
        self.assignment_temperature = float(assignment_temperature)
        self.clustering_iterations = int(clustering_iterations)
        self.variance_prior_strength = float(variance_prior_strength)
        self.min_variance = float(min_variance)
        self.eps = float(eps)

    @staticmethod
    def _class_centroids(
        tokens: torch.Tensor, support_indices: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        centroids = []
        for index in support_indices:
            selected = tokens[index].reshape(-1, tokens.size(-1))
            centroids.append(F.normalize(selected.mean(0), dim=0))
        return torch.stack(centroids)

    @staticmethod
    def _discrimination_margin(
        tokens: torch.Tensor,
        own_class: int,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        similarity = tokens @ centroids.transpose(0, 1)
        own = similarity[:, own_class]
        if centroids.size(0) == 1:
            return own
        other = similarity.clone()
        other[:, own_class] = -torch.inf
        return own - other.max(dim=1).values

    def _descriptor_scores(
        self,
        common_tokens: torch.Tensor,
        polarization_tokens: torch.Tensor,
        support_indices: Sequence[torch.Tensor],
    ):
        common_centroids = self._class_centroids(common_tokens, support_indices)
        polarization_centroids = self._class_centroids(
            polarization_tokens, support_indices
        )
        rows = []
        for class_id, index in enumerate(support_indices):
            common = common_tokens[index]
            polarization = polarization_tokens[index]
            shot_scores = []
            for common_row, polarization_row in zip(common, polarization):
                common_margin = self._discrimination_margin(
                    common_row, class_id, common_centroids
                )
                polarization_margin = self._discrimination_margin(
                    polarization_row, class_id, polarization_centroids
                )
                shot_scores.append(
                    common_margin
                    + self.polarization_selection_weight * polarization_margin
                )
            rows.append(torch.stack(shot_scores))
        return rows

    def _balanced_selection(self, scores: torch.Tensor):
        shot_count, token_count = scores.shape
        if self.training:
            # Keep every support token during training. The continuous quality
            # term below then provides soft selection without a top-k boundary.
            shot_rows = torch.arange(
                shot_count, device=scores.device
            ).repeat_interleave(token_count)
            token_rows = torch.arange(
                token_count, device=scores.device
            ).repeat(shot_count)
            return shot_rows, token_rows, scores.reshape(-1)

        selection_count = min(
            self.selected_descriptor_count, shot_count * token_count
        )
        base_quota = selection_count // shot_count
        remainder = selection_count % shot_count
        if base_quota > token_count:
            raise ValueError("not enough descriptors to satisfy balanced selection")

        quotas = torch.full(
            (shot_count,), base_quota, dtype=torch.long, device=scores.device
        )
        if remainder:
            image_quality = scores.topk(
                k=min(token_count, max(base_quota, 1)), dim=1
            ).values.mean(1)
            extra = image_quality.topk(k=remainder).indices
            quotas[extra] += 1

        shot_rows = []
        token_rows = []
        score_rows = []
        for shot in range(shot_count):
            quota = int(quotas[shot])
            if quota == 0:
                continue
            values, token_index = scores[shot].topk(k=quota)
            shot_rows.append(torch.full_like(token_index, shot))
            token_rows.append(token_index)
            score_rows.append(values)
        return (
            torch.cat(shot_rows),
            torch.cat(token_rows),
            torch.cat(score_rows),
        )

    def _initial_centers(
        self, joint: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        chosen = [int(quality.argmax())]
        while len(chosen) < self.prototype_count:
            centers = joint[torch.tensor(chosen, device=joint.device)]
            nearest = (joint @ centers.transpose(0, 1)).max(dim=1).values
            priority = (1.0 - nearest) * (0.5 + quality)
            priority[torch.tensor(chosen, device=joint.device)] = -torch.inf
            chosen.append(int(priority.argmax()))
        return joint[torch.tensor(chosen, device=joint.device)]

    def _soft_cluster(
        self, joint: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        centers = self._initial_centers(joint, quality)
        weights = None
        for _ in range(self.clustering_iterations):
            similarity = joint @ centers.transpose(0, 1)
            assignment = F.softmax(
                similarity / self.assignment_temperature, dim=1
            )
            weights = assignment * quality.unsqueeze(1)
            denominator = weights.sum(0).clamp_min(self.eps)
            centers = F.normalize(
                weights.transpose(0, 1) @ joint / denominator.unsqueeze(1),
                dim=1,
                eps=self.eps,
            )
        return weights

    def _stream_statistics(
        self, tokens: torch.Tensor, weights: torch.Tensor, quality: torch.Tensor
    ):
        denominator = weights.sum(0).clamp_min(self.eps)
        prototypes = F.normalize(
            weights.transpose(0, 1) @ tokens / denominator.unsqueeze(1),
            dim=1,
            eps=self.eps,
        )
        local_variance = torch.einsum(
            "mk,mkd->kd",
            weights,
            (tokens.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2),
        ) / denominator.unsqueeze(1)

        quality_sum = quality.sum().clamp_min(self.eps)
        global_mean = (quality.unsqueeze(1) * tokens).sum(0) / quality_sum
        global_variance = (
            quality.unsqueeze(1) * (tokens - global_mean).pow(2)
        ).sum(0) / quality_sum
        shrinkage = denominator / (
            denominator + self.variance_prior_strength
        )
        variance = (
            shrinkage.unsqueeze(1) * local_variance
            + (1.0 - shrinkage).unsqueeze(1) * global_variance.unsqueeze(0)
        ).clamp_min(self.min_variance)
        return prototypes, variance, denominator

    def forward(
        self,
        common_tokens: torch.Tensor,
        polarization_tokens: torch.Tensor,
        support_indices: Sequence[torch.Tensor],
    ) -> DualStreamPrototypeBank:
        if common_tokens.shape != polarization_tokens.shape:
            raise ValueError("common and polarization tokens must have equal shapes")
        if common_tokens.ndim != 3:
            raise ValueError("token tensors must have shape [samples, tokens, dim]")
        if not support_indices:
            raise ValueError("support_indices must contain at least one class")

        score_rows = self._descriptor_scores(
            common_tokens, polarization_tokens, support_indices
        )
        common_prototypes = []
        polarization_prototypes = []
        common_variances = []
        polarization_variances = []
        effective_counts = []
        selected_scores = []
        selected_sample_indices = []

        for indices, scores in zip(support_indices, score_rows):
            shot_row, token_row, score = self._balanced_selection(scores)
            sample_row = indices[shot_row]
            common = common_tokens[sample_row, token_row]
            polarization = polarization_tokens[sample_row, token_row]
            joint = F.normalize(
                torch.cat(
                    [
                        common,
                        self.polarization_selection_weight * polarization,
                    ],
                    dim=1,
                ),
                dim=1,
                eps=self.eps,
            )
            quality = torch.sigmoid(score / self.assignment_temperature)
            weights = self._soft_cluster(joint, quality)
            common_proto, common_var, count = self._stream_statistics(
                common, weights, quality
            )
            polarization_proto, polarization_var, _ = self._stream_statistics(
                polarization, weights, quality
            )
            common_prototypes.append(common_proto)
            polarization_prototypes.append(polarization_proto)
            common_variances.append(common_var)
            polarization_variances.append(polarization_var)
            effective_counts.append(count)
            selected_scores.append(score)
            selected_sample_indices.append(sample_row)

        return DualStreamPrototypeBank(
            common_prototypes=torch.stack(common_prototypes),
            polarization_prototypes=torch.stack(polarization_prototypes),
            common_variance=torch.stack(common_variances),
            polarization_variance=torch.stack(polarization_variances),
            effective_count=torch.stack(effective_counts),
            selected_scores=torch.stack(selected_scores),
            selected_sample_indices=torch.stack(selected_sample_indices),
        )
