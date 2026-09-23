from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a formal APEL-RockFSL experiment")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--output_root", default="")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cuda_devices", default="0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def stream(command: list[str], environment: dict[str, str], log_handle) -> None:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
        log_handle.flush()
    code = process.wait()
    if code:
        raise RuntimeError(
            "experiment process failed with exit code {}; see {}".format(
                code, log_handle.name
            )
        )


def main(*, ways: int, shots: int) -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    base_checkpoint = Path(args.base_checkpoint).resolve()
    for split in ("meta_train", "meta_val", "meta_test"):
        if not (data_root / split).is_dir():
            raise SystemExit("Missing {} under data root: {}".format(split, data_root))
    if not base_checkpoint.is_file():
        raise SystemExit("Missing base checkpoint: {}".format(base_checkpoint))

    epochs, warmup, train_tasks = 30, 5, 200
    val_a, val_b, test_tasks = 100, 100, 600
    protocol = "{}way_{}shot".format(ways, shots)
    default_root = PROJECT_ROOT / "outputs" / "formal_{}_seed2026".format(protocol)
    output_root = Path(args.output_root).resolve() if args.output_root else default_root

    train_command = [
        sys.executable,
        "-m",
        "apel_rock_fsl.training",
        "--dataset_root",
        str(data_root),
        "--base_checkpoint",
        str(base_checkpoint),
        "--experiment_root",
        str(output_root),
        "--ways",
        str(ways),
        "--shots",
        str(shots),
        "--queries",
        "10",
        "--epochs",
        str(epochs),
        "--relation_warmup_epochs",
        str(warmup),
        "--train_iterations",
        str(train_tasks),
        "--val_a_iterations",
        str(val_a),
        "--val_b_iterations",
        str(val_b),
        "--manual_seed",
        "2026",
        "--train_episode_seed",
        "12026",
        "--val_a_episode_seed",
        "22026",
        "--val_b_episode_seed",
        "23026",
        "--gate_validation_repeats",
        "3",
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--learning_rate",
        "0.001",
        "--weight_decay",
        "0.0001",
    ]
    test_command = [
        sys.executable,
        "-m",
        "apel_rock_fsl.evaluation",
        "--dataset_root",
        str(data_root),
        "--base_checkpoint",
        str(base_checkpoint),
        "--qacs_checkpoint",
        str(output_root / "best_model.pth"),
        "--output_root",
        str(output_root),
        "--ways",
        str(ways),
        "--shots",
        str(shots),
        "--queries",
        "10",
        "--test_iterations",
        str(test_tasks),
        "--manual_seed",
        "2026",
        "--test_episode_seed",
        "32026",
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--bootstrap_samples",
        "50000",
    ]

    if args.dry_run:
        print("Training command: {}".format(subprocess.list2cmdline(train_command)))
        print("Evaluation command: {}".format(subprocess.list2cmdline(test_command)))
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit("Output directory is not empty: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.device == "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    with (output_root / "experiment.log").open("x", encoding="utf-8") as log_handle:
        stream(train_command, environment, log_handle)
        stream(test_command, environment, log_handle)


if __name__ == "__main__":
    raise SystemExit("Use run_5way_1shot.py or run_5way_5shot.py")
