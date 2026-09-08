# PURLE-RockFSL

PURLE-RockFSL is a few-shot classification implementation for paired plane-polarized light (PPL) and cross-polarized light (XPL) rock thin-section images. This repository contains the final PURLE-ARRC model and the minimum source code needed to train, evaluate, and reproduce the formal 5-way 1-shot and 5-way 5-shot experiments.

## Repository structure

```text
PURLE_RockFSL_GitHub/
├── purle_rock_fsl/          # Model, data pipeline, samplers, training and evaluation
├── train_base_model.py      # Train the frozen SRCF/TDPF/CUPM base model
├── run_5way_1shot.py        # Formal 5-way 1-shot experiment
├── run_5way_5shot.py        # Formal 5-way 5-shot experiment
├── requirements.txt         # Reproducible Python dependencies
└── ENVIRONMENT.md           # Environment installation instructions
```

Historical PCUR modules, quick-test launchers, parameter sweeps, generated outputs, cached files, and local datasets are intentionally excluded from this release.

## Dataset layout

The dataset root must contain three disjoint split directories. Every split contains one directory per rock class:

```text
NJU_Rock_FSL_78_10_20/
├── meta_train/
│   ├── class_001/
│   └── ...
├── meta_val/
└── meta_test/
```

Images in a class are paired by filename. A sample ending in `-k` (`k > 1`) uses the corresponding `-1` image as its paired reference. Supported extensions are JPG, JPEG, PNG, BMP, TIFF, and WebP.

## Environment

Create the environment by following [ENVIRONMENT.md](ENVIRONMENT.md). The verified configuration is Python 3.9.13, PyTorch 2.8.0, TorchVision 0.23.0, and CUDA 12.8.

## Base-model checkpoint

The final PURLE model uses a frozen SRCF/TDPF/CUPM checkpoint. Train one checkpoint for each shot setting from an FGK ResNet-18 encoder checkpoint:

```bash
python train_base_model.py --dataset_root "PATH/TO/NJU_Rock_FSL_78_10_20" --experiment_root "outputs/base_5way_1shot" --encoder_ckpt "PATH/TO/FGK_ENCODER.pth" --classes_per_it_tr 5 --classes_per_it_val 5 --num_support_tr 1 --num_support_val 1
```

For 5-way 5-shot, change both support arguments to `5` and use a separate output directory. The selected checkpoint is saved as `best_model.pth`.

## Formal experiments

Run the 5-way 1-shot experiment:

```bash
python run_5way_1shot.py --data_root "PATH/TO/NJU_Rock_FSL_78_10_20" --base_checkpoint "outputs/base_5way_1shot/best_model.pth"
```

Run the 5-way 5-shot experiment:

```bash
python run_5way_5shot.py --data_root "PATH/TO/NJU_Rock_FSL_78_10_20" --base_checkpoint "outputs/base_5way_5shot/best_model.pth"
```

Useful options:

```text
--device {cuda,cpu}   Execution device; default: cuda
--cuda_devices 0      CUDA device visible to the experiment
--num_workers 8       DataLoader worker count
--output_root PATH    Custom output directory
--dry_run             Validate paths and display commands without training
```

## Console output

Each training epoch prints only the epoch number, training accuracy, and validation accuracy:

```text
Epoch: 1 | Train Accuracy: 95.36% | Validation Accuracy: 94.80%
```

The completed evaluation prints only the final accuracy:

```text
Accuracy: 96.20%
```

Detailed checkpoints, fixed-episode fingerprints, selection records, and query-level diagnostics are still saved silently under `outputs/` for reproducibility. The entire output directory is excluded from Git by default.

The selected final-stage checkpoint is stored as `best_model.pth`; `last_model.pth` stores the unselected final epoch.
