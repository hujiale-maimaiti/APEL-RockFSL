# APEL-RockFSL

## Repository structure

```text
APEL_RockFSL_GitHub/
├── apel_rock_fsl/
├── train_model.py
├── run_5way_1shot.py
├── run_5way_5shot.py
├── requirements.txt
├── DATASET.md
├── ENVIRONMENT.md
└── LICENSE
```

## Dataset layout

```text
NJU_Rock_FSL_78_10_20/
├── meta_train/
│   ├── class_001/
│   └── ...
├── meta_val/
└── meta_test/
```

Images in each class are paired by filename.

## Environment

Create the environment by following [ENVIRONMENT.md](ENVIRONMENT.md). The verified configuration is Python 3.9.13, PyTorch 2.8.0, TorchVision 0.23.0, and CUDA 12.8.
