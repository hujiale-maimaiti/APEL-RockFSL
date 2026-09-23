# coding=utf-8
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ShotAwareReliabilityConstrainedFusion(nn.Module):
    """TAPF/RAPM experts with a bounded, shot-aware cooperative router.

    The fixed D1 shot prior remains the centre of every prediction.  A tiny
    router may only apply a reliability-scaled residual inside a narrow band,
    which prevents an uncertain episode from replacing a stable expert mix.
    """

    def __init__(
        self,
        in_channels=512,
        embedding_dim=128,
        gate_rank=16,
        interaction_rank=16,
        uncertainty_rank=32,
        router_hidden=16,
        min_variance=0.02,
        max_variance=5.0,
        rapm_temperature=10.0,
        branch_loss_weight=0.5,
        router_loss_weight=0.10,
        prior_loss_weight=0.02,
        regret_loss_weight=0.15,
        correction_radius_1shot=0.12,
        correction_radius_multishot=0.10,
        eps=1e-6,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embedding_dim = int(embedding_dim)
        self.min_variance = float(min_variance)
        self.max_variance = float(max_variance)
        self.branch_loss_weight = float(branch_loss_weight)
        self.router_loss_weight = float(router_loss_weight)
        self.prior_loss_weight = float(prior_loss_weight)
        self.regret_loss_weight = float(regret_loss_weight)
        self.correction_radius_1shot = float(correction_radius_1shot)
        self.correction_radius_multishot = float(
            correction_radius_multishot
        )
        self.eps = float(eps)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Separate projections are intentional: RAPM never receives a
        # TAPF-modified embedding, so both standalone expert paths survive.
        self.tapf_projection = nn.Linear(in_channels, embedding_dim)
        self.rapm_projection = nn.Linear(in_channels, embedding_dim)
        for projection in (self.tapf_projection, self.rapm_projection):
            nn.init.trunc_normal_(projection.weight, std=0.02)
            nn.init.zeros_(projection.bias)

        self.task_gate = nn.Sequential(
            nn.Linear(in_channels, gate_rank),
            nn.ReLU(inplace=True),
            nn.Linear(gate_rank, in_channels),
        )
        nn.init.zeros_(self.task_gate[-1].weight)
        nn.init.constant_(self.task_gate[-1].bias, -2.0)
        self.ppl_reduce = nn.Conv2d(
            in_channels, interaction_rank, kernel_size=1, bias=False
        )
        self.xpl_reduce = nn.Conv2d(
            in_channels, interaction_rank, kernel_size=1, bias=False
        )
        self.interaction_expand = nn.Conv2d(
            interaction_rank, in_channels, kernel_size=1
        )
        nn.init.zeros_(self.interaction_expand.weight)
        nn.init.zeros_(self.interaction_expand.bias)
        self.delta_scale = nn.Parameter(torch.tensor(0.1))
        self.interaction_scale = nn.Parameter(torch.tensor(1.0))

        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim, uncertainty_rank),
            nn.GELU(),
            nn.Linear(uncertainty_rank, embedding_dim),
        )
        nn.init.trunc_normal_(self.uncertainty_head[0].weight, std=0.02)
        nn.init.zeros_(self.uncertainty_head[0].bias)
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.constant_(self.uncertainty_head[-1].bias, -2.0)

        self.log_tapf_temperature = nn.Parameter(torch.tensor(0.0))
        self.log_rapm_temperature = nn.Parameter(
            torch.tensor(math.log(rapm_temperature), dtype=torch.float32)
        )

        # Router inputs: two entropy values, two margins, prediction
        # disagreement, JS divergence, query view agreement, support
        # compactness, and normalized shot count.
        self.router = nn.Sequential(
            nn.Linear(9, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, 1),
        )
        nn.init.trunc_normal_(self.router[0].weight, std=0.02)
        nn.init.zeros_(self.router[0].bias)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        self.last_metrics = {}

    @staticmethod
    def _episode_indices(target, n_support):
        classes = torch.unique(target, sorted=True)
        support_indices = []
        query_indices = []
        query_targets = []
        for episode_label, class_id in enumerate(classes):
            indices = torch.nonzero(target == class_id, as_tuple=False).flatten()
            if indices.numel() <= n_support:
                raise ValueError(
                    "Class {} has {} samples; n_support={} leaves no query".format(
                        int(class_id), indices.numel(), n_support
                    )
                )
            support_indices.append(indices[:n_support])
            class_queries = indices[n_support:]
            query_indices.append(class_queries)
            query_targets.append(
                torch.full(
                    (class_queries.numel(),),
                    episode_label,
                    dtype=torch.long,
                    device=target.device,
                )
            )
        return classes, support_indices, query_indices, query_targets

    def _task_fisher_score(self, ppl_map, xpl_map, support_indices):
        ppl = self.pool(ppl_map).flatten(1)
        xpl = self.pool(xpl_map).flatten(1)
        paired_mean = 0.5 * (ppl + xpl)
        class_supports = [paired_mean[index] for index in support_indices]
        class_means = torch.stack([value.mean(0) for value in class_supports])
        within = torch.stack(
            [
                (value - mean.unsqueeze(0)).pow(2).mean(0)
                + 0.25 * (ppl[index] - xpl[index]).pow(2).mean(0)
                for value, mean, index in zip(
                    class_supports, class_means, support_indices
                )
            ]
        ).mean(0)
        between = (class_means - class_means.mean(0, keepdim=True)).pow(2).mean(0)
        fisher = torch.log1p(between / (within + self.eps))
        return F.layer_norm(fisher, (fisher.numel(),))

    def _encode_experts(self, ppl_map, xpl_map, support_indices):
        fisher = self._task_fisher_score(ppl_map, xpl_map, support_indices)
        channel_gate = torch.sigmoid(self.task_gate(fisher)).view(1, -1, 1, 1)
        polarization_delta = channel_gate * (ppl_map - xpl_map)
        interaction = torch.tanh(
            self.ppl_reduce(ppl_map) * self.xpl_reduce(xpl_map)
        )
        interaction = channel_gate * self.interaction_expand(interaction)
        fused_map = (
            xpl_map
            + self.delta_scale * polarization_delta
            + self.interaction_scale * interaction
        )
        tapf_embedding = self.tapf_projection(
            self.pool(fused_map).flatten(1)
        )

        ppl_embedding = self.rapm_projection(self.pool(ppl_map).flatten(1))
        xpl_embedding = self.rapm_projection(self.pool(xpl_map).flatten(1))
        rapm_embedding = 0.5 * (ppl_embedding + xpl_embedding)
        evidence = torch.abs(ppl_embedding - xpl_embedding)
        variance = self.min_variance + F.softplus(
            self.uncertainty_head(evidence)
        )
        variance = variance.clamp(max=self.max_variance)
        return {
            "tapf": tapf_embedding,
            "rapm": rapm_embedding,
            "variance": variance,
            "ppl": ppl_embedding,
            "xpl": xpl_embedding,
            "gate_mean": channel_gate.mean(),
        }

    def _euclidean_logits(self, embedding, support_indices, query_indices):
        prototypes = torch.stack(
            [embedding[index].mean(0) for index in support_indices]
        )
        query_index = torch.cat(query_indices)
        query = embedding[query_index]
        distance = (query.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(-1)
        temperature = self.log_tapf_temperature.exp().clamp(0.25, 20.0)
        return -temperature * distance

    def _uncertainty_logits(
        self, embedding, variance, support_indices, query_indices
    ):
        prototypes = []
        prototype_variances = []
        for index in support_indices:
            support = embedding[index]
            support_variance = variance[index]
            precision = support_variance.reciprocal()
            precision_sum = precision.sum(0).clamp_min(self.eps)
            prototype = (precision * support).sum(0) / precision_sum
            dispersion = (
                precision * (support - prototype.unsqueeze(0)).pow(2)
            ).sum(0) / precision_sum
            observation_variance = (
                precision * support_variance
            ).sum(0) / precision_sum
            class_variance = (dispersion + observation_variance).clamp(
                min=self.min_variance, max=self.max_variance
            )
            prototypes.append(prototype)
            prototype_variances.append(class_variance)
        prototypes = torch.stack(prototypes)
        prototype_variances = torch.stack(prototype_variances)
        query_index = torch.cat(query_indices)
        query = embedding[query_index]
        query_variance = variance[query_index]
        pair_variance = (
            query_variance.unsqueeze(1) + prototype_variances.unsqueeze(0)
        ).clamp(min=self.min_variance, max=2.0 * self.max_variance)
        squared_error = (query.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2)
        distance = 0.5 * (
            squared_error / pair_variance + torch.log(pair_variance)
        ).mean(-1)
        temperature = self.log_rapm_temperature.exp().clamp(1.0, 50.0)
        return -temperature * distance

    @staticmethod
    def _entropy_and_margin(probability):
        class_count = probability.size(1)
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(1) / math.log(max(class_count, 2))
        top2 = probability.topk(k=min(2, class_count), dim=1).values
        margin = top2[:, 0] - top2[:, 1] if class_count > 1 else top2[:, 0]
        return entropy, margin

    def _support_compactness(self, embedding, support_indices):
        values = []
        for index in support_indices:
            support = embedding[index]
            mean = support.mean(0, keepdim=True)
            values.append((support - mean).pow(2).mean())
        compactness = torch.stack(values).mean()
        return torch.log1p(compactness)

    def _router_features(
        self,
        tapf_logits,
        rapm_logits,
        encoded,
        support_indices,
        query_indices,
        n_support,
    ):
        tapf_probability = F.softmax(tapf_logits, dim=1)
        rapm_probability = F.softmax(rapm_logits, dim=1)
        tapf_entropy, tapf_margin = self._entropy_and_margin(tapf_probability)
        rapm_entropy, rapm_margin = self._entropy_and_margin(rapm_probability)
        disagreement = (
            tapf_probability.argmax(1) != rapm_probability.argmax(1)
        ).float()
        mixture = 0.5 * (tapf_probability + rapm_probability)
        js = 0.5 * (
            F.kl_div(mixture.clamp_min(self.eps).log(), tapf_probability, reduction="none").sum(1)
            + F.kl_div(mixture.clamp_min(self.eps).log(), rapm_probability, reduction="none").sum(1)
        )
        query_index = torch.cat(query_indices)
        view_agreement = F.cosine_similarity(
            encoded["ppl"][query_index], encoded["xpl"][query_index], dim=1
        )
        support_compactness = self._support_compactness(
            encoded["rapm"], support_indices
        ).expand_as(view_agreement)
        normalized_shot = view_agreement.new_full(
            view_agreement.shape, min(float(n_support) / 5.0, 1.0)
        )
        return torch.stack(
            [
                tapf_entropy,
                rapm_entropy,
                tapf_margin,
                rapm_margin,
                disagreement,
                js,
                view_agreement,
                support_compactness,
                normalized_shot,
            ],
            dim=1,
        )

    @staticmethod
    def _shot_prior(n_support):
        # TAPF is the stable prior in 1-shot; RAPM receives more weight once
        # several support examples make its class-variance estimate reliable.
        return 0.25 if n_support <= 1 else 0.65

    def _correction_radius(self, n_support):
        if n_support <= 1:
            return self.correction_radius_1shot
        return self.correction_radius_multishot

    @staticmethod
    def _evidence_strength(features):
        """Estimate whether the two experts provide actionable evidence.

        This value is label-free and intentionally conservative.  The router
        moves farther from the prior only when confidence differences, expert
        divergence, and paired-view agreement jointly support a correction.
        """
        entropy_gap = (features[:, 0] - features[:, 1]).abs()
        margin_gap = (features[:, 2] - features[:, 3]).abs()
        js_divergence = features[:, 5].clamp(0.0, 1.0)
        view_agreement = ((features[:, 6] + 1.0) * 0.5).clamp(0.0, 1.0)
        support_quality = torch.exp(-features[:, 7].clamp_min(0.0))
        strength = (
            0.25 * entropy_gap
            + 0.25 * margin_gap
            + 0.20 * js_divergence
            + 0.15 * view_agreement
            + 0.15 * support_quality
        )
        return strength.clamp(0.20, 1.0)

    def forward(self, ppl_map, xpl_map, target, n_support):
        _, support_indices, query_indices, query_targets = self._episode_indices(
            target, n_support
        )
        query_targets = torch.cat(query_targets)
        encoded = self._encode_experts(ppl_map, xpl_map, support_indices)
        tapf_logits = self._euclidean_logits(
            encoded["tapf"], support_indices, query_indices
        )
        rapm_logits = self._uncertainty_logits(
            encoded["rapm"], encoded["variance"], support_indices, query_indices
        )
        tapf_probability = F.softmax(tapf_logits, dim=1)
        rapm_probability = F.softmax(rapm_logits, dim=1)
        features = self._router_features(
            tapf_logits,
            rapm_logits,
            encoded,
            support_indices,
            query_indices,
            n_support,
        )
        prior = features.new_full(
            (features.size(0),), self._shot_prior(n_support)
        )
        radius = self._correction_radius(n_support)
        evidence_strength = self._evidence_strength(features.detach())
        router_residual = torch.tanh(
            self.router(features.detach()).squeeze(1)
        )
        correction = radius * evidence_strength * router_residual
        alpha = (prior + correction).clamp(
            self._shot_prior(n_support) - radius,
            self._shot_prior(n_support) + radius,
        )
        fused_probability = (
            (1.0 - alpha.unsqueeze(1)) * tapf_probability
            + alpha.unsqueeze(1) * rapm_probability
        ).clamp_min(self.eps)
        fused_logits = fused_probability.log()

        fused_loss = F.nll_loss(fused_logits, query_targets)
        tapf_loss = F.cross_entropy(tapf_logits, query_targets)
        rapm_loss = F.cross_entropy(rapm_logits, query_targets)
        tapf_nll = F.cross_entropy(tapf_logits, query_targets, reduction="none")
        rapm_nll = F.cross_entropy(rapm_logits, query_targets, reduction="none")
        preference = torch.tanh(
            (tapf_nll.detach() - rapm_nll.detach()) / 0.25
        )
        route_target = (prior + radius * preference).clamp(
            self._shot_prior(n_support) - radius,
            self._shot_prior(n_support) + radius,
        )
        router_loss = F.smooth_l1_loss(alpha, route_target)
        prior_loss = ((alpha - prior) / max(radius, self.eps)).pow(2).mean()
        fused_nll = F.nll_loss(
            fused_logits, query_targets, reduction="none"
        )
        oracle_branch_nll = torch.minimum(tapf_nll, rapm_nll).detach()
        regret_loss = F.relu(fused_nll - oracle_branch_nll).mean()
        total_loss = (
            fused_loss
            + self.branch_loss_weight * (tapf_loss + rapm_loss)
            + self.router_loss_weight * router_loss
            + self.prior_loss_weight * prior_loss
            + self.regret_loss_weight * regret_loss
        )

        tapf_prediction = tapf_logits.argmax(1)
        rapm_prediction = rapm_logits.argmax(1)
        fused_prediction = fused_logits.argmax(1)
        tapf_correct = tapf_prediction.eq(query_targets)
        rapm_correct = rapm_prediction.eq(query_targets)
        fused_correct = fused_prediction.eq(query_targets)
        self.last_metrics = {
            "loss": total_loss.detach(),
            "fused_loss": fused_loss.detach(),
            "tapf_loss": tapf_loss.detach(),
            "rapm_loss": rapm_loss.detach(),
            "router_loss": router_loss.detach(),
            "prior_loss": prior_loss.detach(),
            "regret_loss": regret_loss.detach(),
            "accuracy": fused_correct.float().mean().detach(),
            "tapf_accuracy": tapf_correct.float().mean().detach(),
            "rapm_accuracy": rapm_correct.float().mean().detach(),
            "oracle_accuracy": (tapf_correct | rapm_correct).float().mean().detach(),
            "disagreement": tapf_prediction.ne(rapm_prediction).float().mean().detach(),
            "router_alpha": alpha.mean().detach(),
            "router_target": route_target.mean().detach(),
            "router_correction": correction.mean().detach(),
            "router_abs_correction": correction.abs().mean().detach(),
            "evidence_strength": evidence_strength.mean().detach(),
            "gate_mean": encoded["gate_mean"].detach(),
            "mean_variance": encoded["variance"].mean().detach(),
            "tapf_temperature": self.log_tapf_temperature.exp().detach(),
            "rapm_temperature": self.log_rapm_temperature.exp().detach(),
        }
        return {
            "loss": total_loss,
            "accuracy": fused_correct.float().mean(),
            "logits": fused_logits,
            "targets": query_targets,
            "tapf_logits": tapf_logits,
            "rapm_logits": rapm_logits,
            "tapf_predictions": tapf_prediction,
            "rapm_predictions": rapm_prediction,
            "fused_predictions": fused_prediction,
            "alpha": alpha,
            "route_target": route_target,
            "metrics": self.last_metrics,
        }

    def parameter_report(self):
        groups = {
            "tapf_projection": self.tapf_projection,
            "rapm_projection": self.rapm_projection,
            "tapf_expert": nn.ModuleList([
                self.task_gate,
                self.ppl_reduce,
                self.xpl_reduce,
                self.interaction_expand,
            ]),
            "rapm_expert": self.uncertainty_head,
            "router": self.router,
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
