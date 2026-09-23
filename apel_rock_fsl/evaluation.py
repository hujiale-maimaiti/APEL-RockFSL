# coding=utf-8
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
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
from .utils import load_checkpoint, seed_everything, seed_worker


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fixed APEL-RockFSL checkpoint")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--qacs_checkpoint", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--test_split", default="meta_test")
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, choices=[1, 5], required=True)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--test_iterations", type=int, default=300)
    parser.add_argument("--manual_seed", type=int, default=2026)
    parser.add_argument("--test_episode_seed", type=int, default=32026)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument(
        "--allow_completed_test_replay",
        action="store_true",
        help=(
            "Replay an already completed fixed meta-test only to export "
            "additional diagnostics. A standard replay must match the saved "
            "episode fingerprint; fixed-class visualization episodes must use "
            "prespecified class ids."
        ),
    )
    parser.add_argument(
        "--visualization_class_ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Prespecified dataset class ids for post-test visualization "
            "episodes. Requires --allow_completed_test_replay."
        ),
    )
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, values.size, size=values.size)
        estimates[index] = values[draw].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def paired_protonet_logits(baseline):
    """Ordinary ProtoNet logits on the paired mean embedding.

    This is the unweighted mean-prototype path used as the B0 visual baseline.
    The positive scale is immaterial to the predicted class, so no additional
    trainable parameter or test-time calibration is introduced here.
    """
    embedding = baseline.rapm_embedding
    prototypes = torch.stack([
        embedding[index].mean(dim=0)
        for index in baseline.support_indices
    ])
    queries = embedding[baseline.flat_query_indices]
    squared_distance = (
        queries.unsqueeze(1) - prototypes.unsqueeze(0)
    ).pow(2).sum(dim=-1)
    return -squared_distance


def enforce_meta_test_policy(
    gate, protocol, checkpoint, allow_completed_test_replay=False
):
    if int(gate.get("validation_bank_count", 0)) < 3:
        raise RuntimeError("QACS go/no-go used fewer than three validation banks")
    if checkpoint.get("checkpoint_role") != "best":
        raise RuntimeError("meta_test requires the frozen selected best checkpoint")
    already_accessed = protocol.get("meta_test_accessed") is True
    if not already_accessed:
        return False
    if not allow_completed_test_replay:
        raise RuntimeError("meta_test has already been accessed for this checkpoint")
    if protocol.get("meta_test_completed") is not True:
        raise RuntimeError(
            "diagnostic replay requires a previously completed meta_test"
        )
    return True


def completed_test_artifact_evidence(run_root, ways, shots):
    paths = {
        "summary": Path(run_root) / "test_summary.json",
        "episodes": Path(run_root) / "test_episodes.csv",
        "queries": Path(run_root) / "test_queries.csv",
    }
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return None
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if int(summary.get("ways", -1)) != int(ways):
        raise RuntimeError("completed test artifact has a different ways value")
    if int(summary.get("shots", -1)) != int(shots):
        raise RuntimeError("completed test artifact has a different shots value")
    iterations = int(
        summary.get("test_iterations", summary.get("test_tasks", -1))
    )
    if iterations <= 0:
        raise RuntimeError("completed test artifact has no valid iteration count")
    fingerprint = summary.get(
        "episode_fingerprint", summary.get("test_episode_fingerprint")
    )
    return {
        "source": "existing_complete_test_artifacts",
        "test_iterations": iterations,
        "episode_fingerprint": fingerprint,
        "summary_path": str(paths["summary"].resolve()),
    }


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    seed_everything(args.manual_seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "summary": output_root / "test_summary.json",
        "episodes": output_root / "test_episodes.csv",
        "queries": output_root / "test_queries.csv",
    }
    existing = [path.name for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite QACS test outputs: " + ", ".join(existing))

    base_path = Path(args.base_checkpoint).resolve()
    qacs_path = Path(args.qacs_checkpoint).resolve()
    checkpoint = load_checkpoint(qacs_path, map_location=device)
    if "qacs" not in checkpoint:
        raise RuntimeError("not a APEL-RockFSL checkpoint")
    if checkpoint.get("base_checkpoint_sha256") != file_sha256(base_path):
        raise RuntimeError("base checkpoint hash does not match QACS training")
    gate_path = qacs_path.parent / "go_no_go.json"
    protocol_path = qacs_path.parent / "protocol.json"
    gate = None
    protocol = None
    completed_test_replay = False
    completion_evidence = None
    if args.test_split == "meta_test":
        if not gate_path.is_file():
            raise RuntimeError("missing go/no-go decision beside QACS checkpoint")
        if not protocol_path.is_file():
            raise RuntimeError("missing protocol.json beside QACS checkpoint")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (
            args.allow_completed_test_replay
            and protocol.get("meta_test_accessed") is True
            and protocol.get("meta_test_completed") is not True
        ):
            completion_evidence = completed_test_artifact_evidence(
                qacs_path.parent, args.ways, args.shots
            )
            if completion_evidence is not None:
                protocol = dict(protocol)
                protocol["meta_test_completed"] = True
        completed_test_replay = enforce_meta_test_policy(
            gate,
            protocol,
            checkpoint,
            args.allow_completed_test_replay,
        )
    trained = checkpoint.get("run_config", {})
    if (
        int(trained.get("ways", args.ways)),
        int(trained.get("shots", args.shots)),
    ) != (args.ways, args.shots):
        raise RuntimeError("QACS checkpoint protocol differs from requested test")

    frozen = load_apel_frozen_srcf(base_path, device=device)
    config = QACSConfig(**checkpoint["qacs_config"])
    model = APELQACSModel(config).to(device)
    model.load_state_dict(checkpoint["qacs"], strict=True)
    model.eval()
    dataset = NJURockSynchronizedPairDataset(
        args.dataset_root,
        args.test_split,
        pair_transform=SynchronizedPairTransform(train=False),
    )
    visualization_class_ids = tuple(args.visualization_class_ids or ())
    visualization_only = bool(visualization_class_ids)
    allowed_indices = None
    if visualization_only:
        if not completed_test_replay:
            raise RuntimeError(
                "fixed-class visualization requires a completed formal test replay"
            )
        if args.test_iterations <= 0:
            raise RuntimeError(
                "fixed-class visualization requires at least one iteration"
            )
        if len(visualization_class_ids) != args.ways:
            raise RuntimeError(
                "visualization_class_ids count must equal the requested ways"
            )
        if len(set(visualization_class_ids)) != len(visualization_class_ids):
            raise RuntimeError("visualization_class_ids must be unique")
        available_classes = {int(value) for value in dataset.y}
        missing_classes = set(visualization_class_ids) - available_classes
        if missing_classes:
            raise RuntimeError(
                "visualization classes are absent from the split: {}".format(
                    sorted(missing_classes)
                )
            )
        selected_classes = set(visualization_class_ids)
        allowed_indices = tuple(
            index
            for index, label in enumerate(dataset.y)
            if int(label) in selected_classes
        )
    sampler = FixedEpisodeBatchSampler(
        dataset.y,
        classes_per_it=args.ways,
        num_samples=args.shots + args.queries,
        iterations=args.test_iterations,
        seed=args.test_episode_seed,
        allowed_indices=allowed_indices,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.test_episode_seed + 100000)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if completed_test_replay and not visualization_only:
        saved_fingerprint = protocol.get("test_episode_fingerprint") or (
            None
            if completion_evidence is None
            else completion_evidence.get("episode_fingerprint")
        )
        saved_iterations = int(
            protocol.get(
                "test_iterations",
                -1
                if completion_evidence is None
                else completion_evidence.get("test_iterations", -1),
            )
        )
        if saved_fingerprint != sampler.fingerprint:
            raise RuntimeError(
                "diagnostic replay episode fingerprint differs from formal test"
            )
        if saved_iterations != args.test_iterations:
            raise RuntimeError(
                "diagnostic replay iteration count differs from formal test"
            )
    elif args.test_split == "meta_test":
        protocol.update({
            "meta_test_accessed": True,
            "meta_test_completed": False,
            "meta_test_accessed_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_episode_fingerprint": sampler.fingerprint,
            "test_iterations": args.test_iterations,
        })
        protocol_path.write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )

    totals = {name: 0 for name in (
        "queries", "protonet_correct", "tapf_correct", "rapm_correct",
        "base_correct", "relation_correct", "qacs_correct",
        "diagnostic_oracle_correct", "rescues", "damages", "beneficial_queries",
        "harmful_queries",
    )}
    episode_rows = []
    query_rows = []
    episode_accuracy = {name: [] for name in (
        "protonet", "tapf", "rapm", "base", "relation", "qacs"
    )}
    with torch.no_grad():
        for episode_id, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = run_apel_episode(
                frozen,
                model,
                images,
                labels,
                args.shots,
                compute_loss=False,
            )
            targets = output.prepared.baseline.query_targets
            protonet_prediction = paired_protonet_logits(
                output.prepared.baseline
            ).argmax(1)
            tapf_prediction = output.prepared.baseline.tapf_logits.argmax(1)
            rapm_prediction = output.prepared.baseline.rapm_logits.argmax(1)
            base_prediction = output.fusion.base_logits.argmax(1)
            relation_prediction = output.fusion.relation_logits.argmax(1)
            qacs_prediction = output.fusion.logits.argmax(1)
            episode_classes = output.prepared.baseline.classes
            true_classes = episode_classes[targets]
            protonet_pred_classes = episode_classes[protonet_prediction]
            tapf_pred_classes = episode_classes[tapf_prediction]
            rapm_pred_classes = episode_classes[rapm_prediction]
            base_pred_classes = episode_classes[base_prediction]
            relation_pred_classes = episode_classes[relation_prediction]
            qacs_pred_classes = episode_classes[qacs_prediction]
            protonet_correct = protonet_prediction.eq(targets)
            tapf_correct = tapf_prediction.eq(targets)
            rapm_correct = rapm_prediction.eq(targets)
            base_correct = base_prediction.eq(targets)
            relation_correct = relation_prediction.eq(targets)
            qacs_correct = qacs_prediction.eq(targets)
            diagnostic_oracle_correct = base_correct | relation_correct
            rescued = ~base_correct & qacs_correct
            damaged = base_correct & ~qacs_correct
            beneficial = ~base_correct & relation_correct
            harmful = base_correct & ~relation_correct
            query_count = int(targets.numel())
            values = {
                "queries": query_count,
                "protonet_correct": int(protonet_correct.sum()),
                "tapf_correct": int(tapf_correct.sum()),
                "rapm_correct": int(rapm_correct.sum()),
                "base_correct": int(base_correct.sum()),
                "relation_correct": int(relation_correct.sum()),
                "qacs_correct": int(qacs_correct.sum()),
                "diagnostic_oracle_correct": int(
                    diagnostic_oracle_correct.sum()
                ),
                "rescues": int(rescued.sum()),
                "damages": int(damaged.sum()),
                "beneficial_queries": int(beneficial.sum()),
                "harmful_queries": int(harmful.sum()),
            }
            for key, value in values.items():
                totals[key] += value
            for name, correct in (
                ("protonet", protonet_correct),
                ("tapf", tapf_correct),
                ("rapm", rapm_correct),
                ("base", base_correct),
                ("relation", relation_correct),
                ("qacs", qacs_correct),
            ):
                episode_accuracy[name].append(float(correct.float().mean()))
            episode_rows.append({
                "episode": episode_id,
                **values,
                "protonet_accuracy": float(protonet_correct.float().mean()),
                "tapf_accuracy": float(tapf_correct.float().mean()),
                "rapm_accuracy": float(rapm_correct.float().mean()),
                "base_accuracy": float(base_correct.float().mean()),
                "relation_accuracy": float(relation_correct.float().mean()),
                "qacs_accuracy": float(qacs_correct.float().mean()),
                "diagnostic_oracle_accuracy": float(
                    diagnostic_oracle_correct.float().mean()
                ),
                "mean_gate": float(output.fusion.gate.mean()),
                "mean_beneficial_gate": float(
                    output.fusion.gate[beneficial].mean()
                    if bool(beneficial.any()) else 0.0
                ),
                "mean_harmful_gate": float(
                    output.fusion.gate[harmful].mean()
                    if bool(harmful.any()) else 0.0
                ),
                "support_advantage": float(output.support_reliability.advantage),
                "support_confidence": float(output.support_reliability.advantage_confidence),
                "support_relation_win_rate": float(output.support_reliability.relation_win_rate),
                "support_reliability_available": int(output.support_reliability.available),
            })
            correction_norm = output.fusion.contrastive_correction.norm(dim=1)
            relation = output.relation.relation
            for query_id in range(query_count):
                query_rows.append({
                    "episode": episode_id,
                    "query": query_id,
                    "target": int(targets[query_id]),
                    "protonet_prediction": int(protonet_prediction[query_id]),
                    "tapf_prediction": int(tapf_prediction[query_id]),
                    "rapm_prediction": int(rapm_prediction[query_id]),
                    "base_prediction": int(base_prediction[query_id]),
                    "relation_prediction": int(relation_prediction[query_id]),
                    "qacs_prediction": int(qacs_prediction[query_id]),
                    "true_class": int(true_classes[query_id]),
                    "protonet_pred_class": int(
                        protonet_pred_classes[query_id]
                    ),
                    "tapf_pred_class": int(tapf_pred_classes[query_id]),
                    "rapm_pred_class": int(rapm_pred_classes[query_id]),
                    "base_pred_class": int(base_pred_classes[query_id]),
                    "relation_pred_class": int(
                        relation_pred_classes[query_id]
                    ),
                    "qacs_pred_class": int(qacs_pred_classes[query_id]),
                    "protonet_correct": int(protonet_correct[query_id]),
                    "tapf_correct": int(tapf_correct[query_id]),
                    "rapm_correct": int(rapm_correct[query_id]),
                    "base_correct": int(base_correct[query_id]),
                    "relation_correct": int(relation_correct[query_id]),
                    "qacs_correct": int(qacs_correct[query_id]),
                    "diagnostic_oracle_correct": int(
                        diagnostic_oracle_correct[query_id]
                    ),
                    "rescued": int(rescued[query_id]),
                    "damaged": int(damaged[query_id]),
                    "gate": float(output.fusion.gate[query_id]),
                    "correction_norm": float(correction_norm[query_id]),
                    "relation_margin": float(relation.margin[query_id]),
                    "relation_entropy": float(relation.entropy[query_id]),
                    "polarization_weight": float(relation.polarization_weight[query_id]),
                    "common_consistency": float(relation.common_consistency[query_id]),
                    "polarization_consistency": float(relation.polarization_consistency[query_id]),
                    "support_advantage": float(output.support_reliability.advantage),
                    "support_confidence": float(output.support_reliability.advantage_confidence),
                    "relation_class_support_advantage": float(
                        output.support_reliability.class_advantage[
                            relation_prediction[query_id]
                        ]
                    ),
                    "relation_class_support_confidence": float(
                        output.support_reliability.class_advantage_confidence[
                            relation_prediction[query_id]
                        ]
                    ),
                    "base_class_support_advantage": float(
                        output.support_reliability.class_advantage[
                            base_prediction[query_id]
                        ]
                    ),
                    "confidence_advantage": float(output.fusion.confidence_advantage[query_id]),
                    "disagreement": int(output.fusion.disagreement[query_id]),
                    "monotonic_score": float(output.fusion.monotonic_score[query_id]),
                })

    query_count = max(totals["queries"], 1)
    accuracy = {
        "protonet": totals["protonet_correct"] / query_count,
        "tapf": totals["tapf_correct"] / query_count,
        "rapm": totals["rapm_correct"] / query_count,
        "base": totals["base_correct"] / query_count,
        "relation": totals["relation_correct"] / query_count,
        "qacs": totals["qacs_correct"] / query_count,
        "diagnostic_oracle_upper_bound": (
            totals["diagnostic_oracle_correct"] / query_count
        ),
    }
    summary = {
        "model": "APEL_ROCK_FSL",
        "display_model": "ProtoNet-TAPF-RAPM-QACS",
        "ways": args.ways,
        "shots": args.shots,
        "queries_per_class": args.queries,
        "test_iterations": args.test_iterations,
        "accuracy": accuracy,
        "confidence_interval_95": {
            name: bootstrap_interval(
                values, args.bootstrap_samples, args.manual_seed + offset
            )
            for offset, (name, values) in enumerate(episode_accuracy.items())
        },
        "paired_net_gain_interval_95": bootstrap_interval(
            np.asarray(episode_accuracy["qacs"])
            - np.asarray(episode_accuracy["base"]),
            args.bootstrap_samples,
            args.manual_seed + 100,
        ),
        "net_gain": (totals["rescues"] - totals["damages"]) / query_count,
        "mean_effective_intervention": float(
            np.mean([row["gate"] for row in query_rows])
        ),
        "episode_fingerprint": sampler.fingerprint,
        "base_checkpoint": str(base_path),
        "qacs_checkpoint": str(qacs_path),
        "go_no_go_passed": None if gate is None else bool(gate.get("passed")),
        "completed_test_replay": completed_test_replay,
        "completed_test_evidence": completion_evidence,
        "visualization_only": visualization_only,
        "visualization_class_ids": list(visualization_class_ids),
        "replay_purpose": (
            "prespecified fixed-class branch export for confusion matrices"
            if visualization_only
            else (
                "fixed-episode branch prediction export for confusion matrices"
                if completed_test_replay else None
            )
        ),
        "qacs_parameters": model.parameter_report(),
        **totals,
    }
    save_csv(output_paths["episodes"], episode_rows)
    save_csv(output_paths["queries"], query_rows)
    output_paths["summary"].write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if args.test_split == "meta_test" and not completed_test_replay:
        protocol.update({
            "meta_test_completed": True,
            "meta_test_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        protocol_path.write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
    print("Accuracy: {:.2f}%".format(100.0 * accuracy["qacs"]), flush=True)


if __name__ == "__main__":
    main()
