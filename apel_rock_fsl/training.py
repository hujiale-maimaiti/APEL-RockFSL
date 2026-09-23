# coding=utf-8
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import (
    QACSConfig,
    NJURockSynchronizedPairDataset,
    APELQACSModel,
    SynchronizedPairTransform,
    load_apel_frozen_srcf,
    run_apel_episode,
)
from .fixed_episode_sampler import FixedEpisodeBatchSampler
from .utils import seed_everything, seed_worker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train APEL-RockFSL around a frozen SRCF checkpoint"
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--train_split", default="meta_train")
    parser.add_argument("--val_split", default="meta_val")
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, choices=[1, 5], required=True)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--relation_warmup_epochs", type=int, default=2)
    parser.add_argument("--train_iterations", type=int, default=100)
    parser.add_argument("--val_a_iterations", type=int, default=50)
    parser.add_argument("--val_b_iterations", type=int, default=50)
    parser.add_argument("--manual_seed", type=int, default=2026)
    parser.add_argument("--train_episode_seed", type=int, default=12026)
    parser.add_argument("--val_a_episode_seed", type=int, default=22026)
    parser.add_argument("--val_b_episode_seed", type=int, default=23026)
    parser.add_argument("--gate_validation_repeats", type=int, default=3)
    parser.add_argument("--minimum_oracle_gap", type=float, default=0.001)
    parser.add_argument("--minimum_pooled_net_gain", type=float, default=0.0)
    parser.add_argument(
        "--minimum_pooled_net_gain_lcb95", type=float, default=0.0
    )
    parser.add_argument(
        "--maximum_damage_rescue_ratio", type=float, default=0.65
    )
    parser.add_argument(
        "--maximum_bank_accuracy_drop", type=float, default=0.0004
    )
    parser.add_argument(
        "--maximum_1shot_accuracy_drop", type=float, default=0.0
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument(
        "--residual_logit_cap",
        type=float,
        default=0.25,
        help="Maximum query-dependent residual added to the QACS gate logit.",
    )
    return parser.parse_args()


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_loader(dataset, sampler, args, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 100000)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    loader.episode_sampler = sampler
    return loader


def build_protocol(args):
    train_dataset = NJURockSynchronizedPairDataset(
        args.dataset_root,
        args.train_split,
        pair_transform=SynchronizedPairTransform(train=True),
    )
    val_dataset = NJURockSynchronizedPairDataset(
        args.dataset_root,
        args.val_split,
        pair_transform=SynchronizedPairTransform(train=False),
    )
    per_episode = args.shots + args.queries
    samplers = {
        "val_a": FixedEpisodeBatchSampler(
            val_dataset.y,
            classes_per_it=args.ways,
            num_samples=per_episode,
            iterations=args.val_a_iterations,
            seed=args.val_a_episode_seed,
        )
    }
    for repeat in range(args.gate_validation_repeats):
        name = "val_gate_{}".format(repeat + 1)
        seed = args.val_b_episode_seed + repeat
        samplers[name] = FixedEpisodeBatchSampler(
            val_dataset.y,
            classes_per_it=args.ways,
            num_samples=per_episode,
            iterations=args.val_b_iterations,
            seed=seed,
        )
    names = list(samplers)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            if set(samplers[left].episodes) & set(samplers[right].episodes):
                raise RuntimeError(
                    "{} and {} contain identical episode manifests".format(
                        left, right
                    )
                )
    protocol = {
        "validation_strategy": "same-class fixed banks with synchronized pair transforms",
        "val_dataset_size": len(val_dataset),
        "val_episode_manifest_overlap": 0,
        "gate_validation_seeds": [
            samplers[name].seed for name in names if name.startswith("val_gate_")
        ],
        "paired_geometry_synchronized": True,
    }
    loaders = {
        name: make_loader(val_dataset, sampler, args, sampler.seed)
        for name, sampler in samplers.items()
    }
    return train_dataset, loaders, protocol


def build_go_no_go(args, validation_metrics):
    failures = []
    checks = []
    if len(validation_metrics) < 3:
        failures.append("fewer than three independent validation banks")
    if args.shots == 5:
        pooled = {
            key: sum(metrics[key] for metrics in validation_metrics.values())
            for key in (
                "queries", "rescues", "damages", "base_correct",
                "oracle_correct",
            )
        }
        for name, metrics in validation_metrics.items():
            oracle_gap = (
                metrics["oracle_base_relation_accuracy"]
                - metrics["base_accuracy"]
            )
            oracle_ok = oracle_gap >= args.minimum_oracle_gap
            accuracy_drop = (
                metrics["base_accuracy"] - metrics["fused_accuracy"]
            )
            bank_ok = accuracy_drop <= args.maximum_bank_accuracy_drop
            checks.append({
                "bank": name,
                "oracle_ok": oracle_ok,
                "oracle_gap": oracle_gap,
                "bank_noninferiority_ok": bank_ok,
                "accuracy_drop": accuracy_drop,
                "metrics": metrics,
            })
            if not oracle_ok:
                failures.append(
                    "{} oracle gap {:.4f} < {:.4f}".format(
                        name, oracle_gap, args.minimum_oracle_gap,
                    )
                )
            if not bank_ok:
                failures.append(
                    "{} accuracy drop {:.4f} > {:.4f}".format(
                        name, accuracy_drop, args.maximum_bank_accuracy_drop,
                    )
                )
        pooled_net_gain = (
            pooled["rescues"] - pooled["damages"]
        ) / max(pooled["queries"], 1)
        pooled_episode_count = sum(
            metrics["episode_count"] for metrics in validation_metrics.values()
        )
        pooled_episode_gain_sum = sum(
            metrics["episode_net_gain_sum"]
            for metrics in validation_metrics.values()
        )
        pooled_episode_gain_square_sum = sum(
            metrics["episode_net_gain_square_sum"]
            for metrics in validation_metrics.values()
        )
        episode_gain_mean = pooled_episode_gain_sum / max(
            pooled_episode_count, 1
        )
        if pooled_episode_count > 1:
            episode_gain_variance = max(
                (
                    pooled_episode_gain_square_sum
                    - pooled_episode_count * episode_gain_mean ** 2
                )
                / (pooled_episode_count - 1),
                0.0,
            )
            net_gain_standard_error = math.sqrt(
                episode_gain_variance / pooled_episode_count
            )
        else:
            net_gain_standard_error = 0.0
        pooled_net_gain_lcb95 = (
            episode_gain_mean - 1.959963984540054 * net_gain_standard_error
        )
        damage_rescue_ratio = pooled["damages"] / max(pooled["rescues"], 1)
        oracle_opportunities = pooled["oracle_correct"] - pooled["base_correct"]
        capture_ratio = (
            (pooled["rescues"] - pooled["damages"])
            / max(oracle_opportunities, 1)
        )
        if pooled_net_gain <= args.minimum_pooled_net_gain:
            failures.append(
                "pooled net gain {:.4f} <= {:.4f}".format(
                    pooled_net_gain, args.minimum_pooled_net_gain
                )
            )
        if pooled_net_gain_lcb95 <= args.minimum_pooled_net_gain_lcb95:
            failures.append(
                "pooled net-gain LCB95 {:.4f} <= {:.4f}".format(
                    pooled_net_gain_lcb95,
                    args.minimum_pooled_net_gain_lcb95,
                )
            )
        if damage_rescue_ratio > args.maximum_damage_rescue_ratio:
            failures.append(
                "damage/rescue ratio {:.4f} > {:.4f}".format(
                    damage_rescue_ratio, args.maximum_damage_rescue_ratio
                )
            )
        pooled_metrics = {
            **pooled,
            "net_gain": pooled_net_gain,
            "net_gain_standard_error": net_gain_standard_error,
            "net_gain_lcb95": pooled_net_gain_lcb95,
            "episode_count": pooled_episode_count,
            "episode_net_gain_mean": episode_gain_mean,
            "damage_rescue_ratio": damage_rescue_ratio,
            "oracle_opportunities": oracle_opportunities,
            "oracle_capture_ratio": capture_ratio,
        }
    else:
        pooled_metrics = None
        for name, metrics in validation_metrics.items():
            accuracy_drop = (
                metrics["base_accuracy"] - metrics["fused_accuracy"]
            )
            noninferior = accuracy_drop <= args.maximum_1shot_accuracy_drop
            checks.append({
                "bank": name,
                "noninferiority_ok": noninferior,
                "accuracy_drop": accuracy_drop,
                "metrics": metrics,
            })
            if not noninferior:
                failures.append(
                    "{} accuracy drop {:.4f} > {:.4f}".format(
                        name,
                        accuracy_drop,
                        args.maximum_1shot_accuracy_drop,
                    )
                )
    return {
        "passed": not failures,
        "shots": args.shots,
        "validation_bank_count": len(validation_metrics),
        "criteria": {
            "minimum_oracle_gap": args.minimum_oracle_gap,
            "minimum_pooled_net_gain": args.minimum_pooled_net_gain,
            "minimum_pooled_net_gain_lcb95": args.minimum_pooled_net_gain_lcb95,
            "maximum_damage_rescue_ratio": args.maximum_damage_rescue_ratio,
            "maximum_bank_accuracy_drop": args.maximum_bank_accuracy_drop,
            "maximum_1shot_accuracy_drop": args.maximum_1shot_accuracy_drop,
            "all_validation_banks_must_pass": True,
        },
        "pooled_metrics": pooled_metrics,
        "checks": checks,
        "failures": failures,
    }


def build_train_loader(dataset, args, epoch):
    seed = args.train_episode_seed + epoch - 1
    sampler = FixedEpisodeBatchSampler(
        dataset.y,
        classes_per_it=args.ways,
        num_samples=args.shots + args.queries,
        iterations=args.train_iterations,
        seed=seed,
    )
    return make_loader(dataset, sampler, args, seed)


def summarize_episode(output):
    targets = output.prepared.baseline.query_targets
    base = output.fusion.base_logits.argmax(1)
    relation = output.fusion.relation_logits.argmax(1)
    fused = output.fusion.logits.argmax(1)
    base_correct = base.eq(targets)
    relation_correct = relation.eq(targets)
    fused_correct = fused.eq(targets)
    beneficial = ~base_correct & relation_correct
    harmful = base_correct & ~relation_correct
    gate = output.fusion.gate.detach()
    query_count = int(targets.numel())
    rescues = int((~base_correct & fused_correct).sum())
    damages = int((base_correct & ~fused_correct).sum())
    episode_net_gain = (rescues - damages) / max(query_count, 1)
    return {
        "queries": query_count,
        "base_correct": int(base_correct.sum()),
        "relation_correct": int(relation_correct.sum()),
        "fused_correct": int(fused_correct.sum()),
        "oracle_correct": int((base_correct | relation_correct).sum()),
        "rescues": rescues,
        "damages": damages,
        "beneficial_queries": int(beneficial.sum()),
        "harmful_queries": int(harmful.sum()),
        "rescuable_queries": int(output.loss.rescuable_count),
        "gate_sum": float(gate.sum()),
        "beneficial_gate_sum": float((gate * beneficial.float()).sum()),
        "harmful_gate_sum": float((gate * harmful.float()).sum()),
        "relation_margin_sum": float(output.relation.relation.margin.detach().sum()),
        "relation_entropy_sum": float(output.relation.relation.entropy.detach().sum()),
        "support_advantage": float(output.support_reliability.advantage),
        "support_confidence": float(output.support_reliability.advantage_confidence),
        "gate_supervision_loss_sum": float(output.loss.gate_supervision_loss),
        "gate_rescue_loss_sum": float(output.loss.gate_rescue_loss),
        "gate_damage_loss_sum": float(output.loss.gate_damage_loss),
        "gate_unproductive_loss_sum": float(
            output.loss.gate_unproductive_loss
        ),
        "counterfactual_target_sum": float(
            output.loss.counterfactual_target_mean
            * output.loss.rescuable_count
        ),
        "episode_net_gain_sum": episode_net_gain,
        "episode_net_gain_square_sum": episode_net_gain ** 2,
    }


def finalize_metrics(losses, totals, episode_count):
    query_count = max(totals["queries"], 1)
    beneficial_count = totals["beneficial_queries"]
    harmful_count = totals["harmful_queries"]
    rescuable_count = totals["rescuable_queries"]
    episode_gain_mean = totals["episode_net_gain_sum"] / episode_count
    if episode_count > 1:
        episode_gain_variance = max(
            (
                totals["episode_net_gain_square_sum"]
                - episode_count * episode_gain_mean ** 2
            )
            / (episode_count - 1),
            0.0,
        )
        episode_gain_standard_error = math.sqrt(
            episode_gain_variance / episode_count
        )
    else:
        episode_gain_standard_error = 0.0
    beneficial_gate = (
        totals["beneficial_gate_sum"] / beneficial_count
        if beneficial_count else 0.0
    )
    harmful_gate = (
        totals["harmful_gate_sum"] / harmful_count
        if harmful_count else 0.0
    )
    return {
        "loss": float(np.mean(losses)),
        "base_accuracy": totals["base_correct"] / query_count,
        "relation_accuracy": totals["relation_correct"] / query_count,
        "fused_accuracy": totals["fused_correct"] / query_count,
        "oracle_base_relation_accuracy": totals["oracle_correct"] / query_count,
        "net_gain": (totals["rescues"] - totals["damages"]) / query_count,
        "net_gain_lcb95": (
            episode_gain_mean
            - 1.959963984540054 * episode_gain_standard_error
        ),
        "episode_count": episode_count,
        "episode_net_gain_sum": totals["episode_net_gain_sum"],
        "episode_net_gain_square_sum": totals["episode_net_gain_square_sum"],
        "rescue_rate": totals["rescues"] / query_count,
        "damage_rate": totals["damages"] / query_count,
        "mean_gate": totals["gate_sum"] / query_count,
        "mean_beneficial_gate": beneficial_gate,
        "mean_harmful_gate": harmful_gate,
        "gate_separation": beneficial_gate - harmful_gate,
        "mean_relation_margin": totals["relation_margin_sum"] / query_count,
        "mean_relation_entropy": totals["relation_entropy_sum"] / query_count,
        "mean_support_advantage": totals["support_advantage"] / episode_count,
        "mean_support_confidence": totals["support_confidence"] / episode_count,
        "mean_gate_supervision_loss": totals["gate_supervision_loss_sum"] / episode_count,
        "mean_gate_rescue_loss": totals["gate_rescue_loss_sum"] / episode_count,
        "mean_gate_damage_loss": totals["gate_damage_loss_sum"] / episode_count,
        "mean_gate_unproductive_loss": (
            totals["gate_unproductive_loss_sum"] / episode_count
        ),
        "mean_counterfactual_target": (
            totals["counterfactual_target_sum"] / rescuable_count
            if rescuable_count else 0.0
        ),
        **{key: totals[key] for key in (
            "queries", "base_correct", "relation_correct", "fused_correct",
            "oracle_correct", "rescues", "damages", "beneficial_queries",
            "harmful_queries", "rescuable_queries",
        )},
    }


@torch.no_grad()
def evaluate(loader, frozen, model, device, shots, description):
    model.eval()
    losses = []
    totals = {key: 0 for key in (
        "queries", "base_correct", "relation_correct", "fused_correct",
        "oracle_correct", "rescues", "damages", "beneficial_queries",
        "harmful_queries", "rescuable_queries",
    )}
    totals.update({key: 0.0 for key in (
        "gate_sum", "beneficial_gate_sum", "harmful_gate_sum",
        "relation_margin_sum", "relation_entropy_sum",
        "support_advantage", "support_confidence",
        "gate_supervision_loss_sum", "gate_rescue_loss_sum",
        "gate_damage_loss_sum", "gate_unproductive_loss_sum",
        "counterfactual_target_sum", "episode_net_gain_sum",
        "episode_net_gain_square_sum",
    )})
    episodes = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = run_apel_episode(frozen, model, images, labels, shots)
        losses.append(float(output.loss.loss))
        for key, value in summarize_episode(output).items():
            totals[key] += value
        episodes += 1
    return finalize_metrics(losses, totals, max(episodes, 1))


def state_to_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.gate_validation_repeats < 3:
        raise ValueError("gate_validation_repeats must be at least 3")
    if not 0 <= args.relation_warmup_epochs < args.epochs:
        raise ValueError(
            "relation_warmup_epochs must be non-negative and lower than epochs"
        )
    if args.residual_logit_cap < 0.0:
        raise ValueError("residual_logit_cap must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    seed_everything(args.manual_seed)
    root = Path(args.experiment_root)
    expected = [root / name for name in (
        "best_model.pth", "last_model.pth", "training_history.csv",
        "model_selection.json", "protocol.json", "support_reliability.json",
        "go_no_go.json",
    )]
    existing = [path.name for path in expected if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite APEL outputs: " + ", ".join(existing))
    root.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_checkpoint).resolve()
    frozen = load_apel_frozen_srcf(base_path, device=device)
    trained = frozen.checkpoint.get("config", {})
    trained_protocol = (
        int(trained.get("classes_per_it_val", args.ways)),
        int(trained.get("num_support_val", args.shots)),
    )
    if trained_protocol != (args.ways, args.shots):
        raise RuntimeError(
            "base checkpoint protocol is {}w{}s, requested {}w{}s".format(
                *trained_protocol, args.ways, args.shots
            )
        )

    config = QACSConfig(residual_logit_cap=args.residual_logit_cap)
    model = APELQACSModel(config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_dataset, val_loaders, protocol_info = build_protocol(args)
    history = []
    best = None
    last_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = build_train_loader(train_dataset, args, epoch)
        warmup = epoch <= args.relation_warmup_epochs
        train_losses = []
        totals = {key: 0 for key in (
            "queries", "base_correct", "relation_correct", "fused_correct",
            "oracle_correct", "rescues", "damages", "beneficial_queries",
            "harmful_queries", "rescuable_queries",
        )}
        totals.update({key: 0.0 for key in (
            "gate_sum", "beneficial_gate_sum", "harmful_gate_sum",
            "relation_margin_sum", "relation_entropy_sum",
            "support_advantage", "support_confidence",
            "gate_supervision_loss_sum", "gate_rescue_loss_sum",
            "gate_damage_loss_sum", "gate_unproductive_loss_sum",
            "counterfactual_target_sum", "episode_net_gain_sum",
            "episode_net_gain_square_sum",
        )})
        episodes = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = run_apel_episode(
                frozen,
                model,
                images,
                labels,
                args.shots,
                compute_support_reliability=not warmup,
            )
            if warmup:
                loss = F.cross_entropy(
                    output.relation.relation.logits,
                    output.prepared.baseline.query_targets,
                )
            else:
                loss = output.loss.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            for key, value in summarize_episode(output).items():
                totals[key] += value
            episodes += 1
        train_metrics = finalize_metrics(
            train_losses, totals, max(episodes, 1)
        )
        val_metrics = evaluate(
            val_loaders["val_a"],
            frozen,
            model,
            device,
            args.shots,
            "APEL val-A",
        )
        row = {
            "epoch": epoch,
            "phase": "relation_warmup" if warmup else "joint_fusion",
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_a_{key}": value for key, value in val_metrics.items()},
            "train_episode_fingerprint": loader.episode_sampler.fingerprint,
            "val_a_episode_fingerprint": val_loaders["val_a"].episode_sampler.fingerprint,
        }
        history.append(row)
        state = state_to_cpu(model)
        last_state = state
        if not warmup:
            candidate = (
                val_metrics["net_gain_lcb95"],
                val_metrics["net_gain"],
                -val_metrics["damage_rate"],
                val_metrics["gate_separation"],
                -val_metrics["loss"],
            )
            if best is None or candidate > best[0]:
                best = (candidate, epoch, state, val_metrics)
        print(
            "Epoch: {} | Train Accuracy: {:.2f}% | Validation Accuracy: {:.2f}%".format(
                epoch,
                100.0 * train_metrics["fused_accuracy"],
                100.0 * val_metrics["fused_accuracy"],
            ),
            flush=True,
        )

    _, best_epoch, best_state, best_metrics = best
    model.load_state_dict(best_state, strict=True)
    gate_validation_metrics = {
        name: evaluate(
            loader,
            frozen,
            model,
            device,
            args.shots,
            "APEL {}".format(name),
        )
        for name, loader in val_loaders.items()
        if name.startswith("val_gate_")
    }
    go_no_go = build_go_no_go(args, gate_validation_metrics)
    base_hash = file_sha256(base_path)
    payload = {
        "model": "APEL_ROCK_FSL",
        "checkpoint_role": "best",
        "qacs": best_state,
        "qacs_config": asdict(config),
        "run_config": vars(args),
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_hash,
        "selected_epoch": best_epoch,
        "val_a_metrics": best_metrics,
        "gate_validation_metrics": gate_validation_metrics,
        "go_no_go_passed": go_no_go["passed"],
    }
    torch.save(payload, root / "best_model.pth")
    torch.save({
        "model": "APEL_ROCK_FSL",
        "checkpoint_role": "last_unselected",
        "qacs": last_state,
        "qacs_config": asdict(config),
        "run_config": vars(args),
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_hash,
        "last_epoch": args.epochs,
    }, root / "last_model.pth")
    save_csv(root / "training_history.csv", history)
    (root / "model_selection.json").write_text(json.dumps({
        "selected_epoch": best_epoch,
        "criterion": "val-A net-gain LCB95, net gain, lower damage, gate separation, lower loss",
        "val_a_metrics": best_metrics,
    }, indent=2), encoding="utf-8")
    (root / "support_reliability.json").write_text(json.dumps({
        "strategy": (
            "balanced leave-one-shot-out episode- and class-level "
            "continuous NLL"
        ),
        "leakage_control": "fold-specific TAPF and APEL support-conditioned encoding",
        "gate_validation_metrics": gate_validation_metrics,
    }, indent=2), encoding="utf-8")
    (root / "go_no_go.json").write_text(
        json.dumps(go_no_go, indent=2), encoding="utf-8"
    )
    (root / "protocol.json").write_text(json.dumps({
        "val_a_episode_fingerprint": val_loaders["val_a"].episode_sampler.fingerprint,
        "gate_validation_episode_fingerprints": {
            name: loader.episode_sampler.fingerprint
            for name, loader in val_loaders.items()
            if name.startswith("val_gate_")
        },
        "base_checkpoint_sha256": base_hash,
        "meta_test_accessed": False,
        "meta_test_completed": False,
        **protocol_info,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
