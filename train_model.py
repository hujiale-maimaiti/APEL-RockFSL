# coding=utf-8
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from paum_rock_fsl.base_fusion_model import ShotAwareReliabilityConstrainedFusion
from paum_rock_fsl.dataset import NJURockPairDataset
from paum_rock_fsl.deterministic_episode_sampler import DeterministicEpisodeSampler
from paum_rock_fsl.resnet_backbone import (
    FGKMiniResNet18Backbone,
    load_fgk_resnet18_backbone,
)


TRANSFORMS = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deterministic shot-aware reliability-constrained fusion"
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--train_split", default="meta_train")
    parser.add_argument("--val_split", default="meta_val")
    parser.add_argument("--test_split", default="meta_test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--val_iterations", type=int, default=100)
    parser.add_argument("--test_iterations", type=int, default=600)
    parser.add_argument("--classes_per_it_tr", type=int, default=5)
    parser.add_argument("--num_support_tr", type=int, default=1)
    parser.add_argument("--num_query_tr", type=int, default=10)
    parser.add_argument("--classes_per_it_val", type=int, default=5)
    parser.add_argument("--num_support_val", type=int, default=1)
    parser.add_argument("--num_query_val", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--module_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_step", type=int, default=15)
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.3)
    parser.add_argument("--manual_seed", type=int, default=2026)
    parser.add_argument("--train_episode_seed", type=int, default=12026)
    parser.add_argument("--val_episode_seed", type=int, default=22026)
    parser.add_argument("--test_episode_seed", type=int, default=32026)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--branch_loss_weight", type=float, default=0.5)
    parser.add_argument("--router_loss_weight", type=float, default=0.10)
    parser.add_argument("--prior_loss_weight", type=float, default=0.02)
    parser.add_argument("--regret_loss_weight", type=float, default=0.15)
    parser.add_argument("--correction_radius_1shot", type=float, default=0.12)
    parser.add_argument(
        "--correction_radius_multishot", type=float, default=0.10
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def split_episode_seed(args, mode):
    return {
        "train": args.train_episode_seed,
        "val": args.val_episode_seed,
        "test": args.test_episode_seed,
    }[mode]


def build_loader(args, mode):
    split = {
        "train": args.train_split,
        "val": args.val_split,
        "test": args.test_split,
    }[mode]
    dataset = NJURockPairDataset(
        root=args.dataset_root,
        split=split,
        transform=TRANSFORMS[mode],
    )
    ways = args.classes_per_it_tr if mode == "train" else args.classes_per_it_val
    shots = args.num_support_tr if mode == "train" else args.num_support_val
    queries = args.num_query_tr if mode == "train" else args.num_query_val
    tasks = {
        "train": args.iterations,
        "val": args.val_iterations,
        "test": args.test_iterations,
    }[mode]
    episode_seed = split_episode_seed(args, mode)
    sampler = DeterministicEpisodeSampler(
        labels=dataset.y,
        classes_per_it=ways,
        num_samples=shots + queries,
        iterations=tasks,
        seed=episode_seed,
        record_episodes=mode == "test",
    )
    worker_generator = torch.Generator(device="cpu")
    worker_generator.manual_seed(episode_seed + 100000)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        worker_init_fn=seed_worker,
        generator=worker_generator,
    )
    loader.episode_sampler = sampler
    return loader


def build_models(args, device):
    backbone = FGKMiniResNet18Backbone()
    checkpoint = load_checkpoint(args.encoder_ckpt, map_location=device)
    load_fgk_resnet18_backbone(backbone, checkpoint, strict=False)
    if args.freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    backbone.to(device)

    module = ShotAwareReliabilityConstrainedFusion(
        in_channels=512,
        embedding_dim=128,
        gate_rank=16,
        interaction_rank=16,
        uncertainty_rank=32,
        branch_loss_weight=args.branch_loss_weight,
        router_loss_weight=args.router_loss_weight,
        prior_loss_weight=args.prior_loss_weight,
        regret_loss_weight=args.regret_loss_weight,
        correction_radius_1shot=args.correction_radius_1shot,
        correction_radius_multishot=args.correction_radius_multishot,
    ).to(device)
    return backbone, module


def run_episode(batch, backbone, module, device, n_support):
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    ppl_map = backbone(images[:, 3:6])
    xpl_map = backbone(images[:, 0:3])
    return module(ppl_map, xpl_map, target=labels, n_support=n_support)


def evaluate(loader, backbone, module, device, n_support, description):
    backbone.eval()
    module.eval()
    losses = []
    accuracies = []
    diagnostics = []
    with torch.no_grad():
        for episode, batch in enumerate(loader, start=1):
            result = run_episode(
                batch, backbone, module, device, n_support
            )
            losses.append(float(result["loss"].item()))
            accuracies.append(float(result["accuracy"].item()))
            targets = result["targets"].detach().cpu().tolist()
            tdpf_predictions = result["tdpf_predictions"].detach().cpu().tolist()
            cupm_predictions = result["cupm_predictions"].detach().cpu().tolist()
            fused_predictions = result["fused_predictions"].detach().cpu().tolist()
            alphas = result["alpha"].detach().cpu().tolist()
            for query, values in enumerate(
                zip(
                    targets,
                    tdpf_predictions,
                    cupm_predictions,
                    fused_predictions,
                    alphas,
                )
            ):
                target, tdpf_prediction, cupm_prediction, fused_prediction, alpha = values
                diagnostics.append({
                    "episode": episode,
                    "query": query,
                    "target": target,
                    "tdpf_prediction": tdpf_prediction,
                    "cupm_prediction": cupm_prediction,
                    "fused_prediction": fused_prediction,
                    "tdpf_correct": int(tdpf_prediction == target),
                    "cupm_correct": int(cupm_prediction == target),
                    "fused_correct": int(fused_prediction == target),
                    "oracle_correct": int(
                        tdpf_prediction == target or cupm_prediction == target
                    ),
                    "experts_disagree": int(tdpf_prediction != cupm_prediction),
                    "cupm_weight": alpha,
                })
    return (
        float(np.mean(losses)),
        float(np.mean(accuracies)),
        accuracies,
        diagnostics,
    )


def save_state(path, backbone, module, epoch, val_accuracy, args):
    torch.save({
        "backbone": backbone.state_dict(),
        "srcf_module": module.state_dict(),
        "model": "SRCF_TDPF_CUPM",
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "config": vars(args),
    }, path)


def metric_value(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def train(args, loaders, backbone, module, device, experiment_root):
    backbone_parameters = [
        parameter for parameter in backbone.parameters() if parameter.requires_grad
    ]
    backbone_optimizer = None
    backbone_scheduler = None
    if backbone_parameters:
        backbone_optimizer = optim.Adam(
            backbone_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        backbone_scheduler = optim.lr_scheduler.StepLR(
            backbone_optimizer,
            step_size=args.lr_scheduler_step,
            gamma=args.lr_scheduler_gamma,
        )

    module_parameters = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    module_optimizer = optim.Adam(
        module_parameters,
        lr=args.module_learning_rate,
        weight_decay=args.weight_decay,
    )
    module_scheduler = optim.lr_scheduler.StepLR(
        module_optimizer,
        step_size=args.lr_scheduler_step,
        gamma=args.lr_scheduler_gamma,
    )

    best_accuracy = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        backbone.train(not args.freeze_backbone)
        module.train()
        train_losses = []
        train_accuracies = []
        iterator = loaders["train"]
        for batch in iterator:
            if backbone_optimizer is not None:
                backbone_optimizer.zero_grad()
            module_optimizer.zero_grad()
            result = run_episode(
                batch, backbone, module, device, args.num_support_tr
            )
            result["loss"].backward()
            if backbone_optimizer is not None:
                backbone_optimizer.step()
            module_optimizer.step()
            train_losses.append(float(result["loss"].item()))
            train_accuracies.append(float(result["accuracy"].item()))

        if backbone_scheduler is not None:
            backbone_scheduler.step()
        module_scheduler.step()
        val_loss, val_accuracy, _, val_diagnostics = evaluate(
            loaders["val"],
            backbone,
            module,
            device,
            args.num_support_val,
            "validation",
        )
        row = {
            "epoch": epoch,
            "model": "SRCF_TDPF_CUPM",
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": float(np.mean(train_accuracies)),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_tdpf_accuracy": float(np.mean([
                item["tdpf_correct"] for item in val_diagnostics
            ])),
            "val_cupm_accuracy": float(np.mean([
                item["cupm_correct"] for item in val_diagnostics
            ])),
            "val_oracle_accuracy": float(np.mean([
                item["oracle_correct"] for item in val_diagnostics
            ])),
            "router_alpha": metric_value(module.last_metrics["router_alpha"]),
            "router_loss": metric_value(module.last_metrics["router_loss"]),
            "prior_loss": metric_value(module.last_metrics["prior_loss"]),
            "regret_loss": metric_value(module.last_metrics["regret_loss"]),
            "router_abs_correction": metric_value(
                module.last_metrics["router_abs_correction"]
            ),
            "evidence_strength": metric_value(
                module.last_metrics["evidence_strength"]
            ),
            "gate_mean": metric_value(module.last_metrics["gate_mean"]),
            "mean_variance": metric_value(module.last_metrics["mean_variance"]),
            "train_episode_fingerprint": loaders["train"].episode_sampler.fingerprint,
            "val_episode_fingerprint": loaders["val"].episode_sampler.fingerprint,
        }
        history.append(row)
        print(
            "Epoch: {} | Train Accuracy: {:.2f}% | Validation Accuracy: {:.2f}%".format(
                epoch,
                100.0 * row["train_accuracy"],
                100.0 * val_accuracy,
            ),
            flush=True,
        )
        save_state(
            experiment_root / "last_model.pth",
            backbone,
            module,
            epoch,
            val_accuracy,
            args,
        )
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            best_state = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "srcf_module": copy.deepcopy(module.state_dict()),
            }
            save_state(
                experiment_root / "best_model.pth",
                backbone,
                module,
                epoch,
                val_accuracy,
                args,
            )

    with (experiment_root / "train_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    backbone.load_state_dict(best_state["backbone"])
    module.load_state_dict(best_state["srcf_module"])
    return best_accuracy


def save_test_results(
    experiment_root,
    accuracies,
    diagnostics,
    mean_accuracy,
    ci95,
    test_sampler,
    args,
    module,
    train_episode_fingerprint,
    val_episode_fingerprint,
):
    records = test_sampler.episode_records
    if len(records) != len(accuracies):
        raise RuntimeError(
            "Recorded {} test episodes but evaluated {}".format(
                len(records), len(accuracies)
            )
        )
    with (experiment_root / "test_episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "accuracy", "sample_indices"])
        for episode, (accuracy, indices) in enumerate(
            zip(accuracies, records), start=1
        ):
            writer.writerow([
                episode,
                accuracy,
                ";".join(str(index) for index in indices),
            ])

    with (experiment_root / "test_queries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)

    complementarity = {
        "tdpf_accuracy": float(np.mean([
            item["tdpf_correct"] for item in diagnostics
        ])),
        "cupm_accuracy": float(np.mean([
            item["cupm_correct"] for item in diagnostics
        ])),
        "fused_accuracy_query_level": float(np.mean([
            item["fused_correct"] for item in diagnostics
        ])),
        "oracle_accuracy": float(np.mean([
            item["oracle_correct"] for item in diagnostics
        ])),
        "disagreement_rate": float(np.mean([
            item["experts_disagree"] for item in diagnostics
        ])),
        "tdpf_only_correct": float(np.mean([
            item["tdpf_correct"] and not item["cupm_correct"]
            for item in diagnostics
        ])),
        "cupm_only_correct": float(np.mean([
            item["cupm_correct"] and not item["tdpf_correct"]
            for item in diagnostics
        ])),
        "mean_cupm_weight": float(np.mean([
            item["cupm_weight"] for item in diagnostics
        ])),
    }

    protocol = {
        "model": "SRCF_TDPF_CUPM",
        "branch_loss_weight": args.branch_loss_weight,
        "router_loss_weight": args.router_loss_weight,
        "prior_loss_weight": args.prior_loss_weight,
        "regret_loss_weight": args.regret_loss_weight,
        "correction_radius_1shot": args.correction_radius_1shot,
        "correction_radius_multishot": args.correction_radius_multishot,
        "epochs": args.epochs,
        "train_tasks_per_epoch": args.iterations,
        "validation_tasks_per_epoch": args.val_iterations,
        "test_tasks": args.test_iterations,
        "ways": args.classes_per_it_val,
        "shots": args.num_support_val,
        "queries": args.num_query_val,
        "manual_seed": args.manual_seed,
        "train_episode_seed": args.train_episode_seed,
        "val_episode_seed": args.val_episode_seed,
        "test_episode_seed": args.test_episode_seed,
        "train_episode_fingerprint": train_episode_fingerprint,
        "val_episode_fingerprint": val_episode_fingerprint,
        "test_episode_fingerprint": test_sampler.fingerprint,
    }
    summary = {
        **protocol,
        "episodes": len(accuracies),
        "mean_accuracy": mean_accuracy,
        "mean_accuracy_percent": mean_accuracy * 100.0,
        "ci95": ci95,
        "ci95_percent": ci95 * 100.0,
        "parameter_report": module.parameter_report(),
        "complementarity": complementarity,
    }
    with (experiment_root / "test_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False"
        )
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    experiment_root = Path(args.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    seed_everything(args.manual_seed)
    loaders = {
        mode: build_loader(args, mode) for mode in ("train", "val")
    }
    backbone, module = build_models(args, device)
    best_val = train(
        args, loaders, backbone, module, device, experiment_root
    )

    # Test sampling is independent of training duration and model RNG use.
    test_loader = build_loader(args, "test")
    test_loss, test_accuracy, episode_accuracies, diagnostics = evaluate(
        test_loader,
        backbone,
        module,
        device,
        args.num_support_val,
        "meta-test",
    )
    count = len(episode_accuracies)
    std = float(np.std(episode_accuracies, ddof=1)) if count > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(count) if count > 1 else 0.0
    save_test_results(
        experiment_root,
        episode_accuracies,
        diagnostics,
        test_accuracy,
        ci95,
        test_loader.episode_sampler,
        args,
        module,
        loaders["train"].episode_sampler.fingerprint,
        loaders["val"].episode_sampler.fingerprint,
    )
    print("Accuracy: {:.2f}%".format(test_accuracy * 100.0), flush=True)


if __name__ == "__main__":
    main()
