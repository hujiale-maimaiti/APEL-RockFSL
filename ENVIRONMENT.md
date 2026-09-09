# Environment Setup

The code was verified with the following local environment:

- Windows 10/11
- Python 3.9.13
- PyTorch 2.8.0 with CUDA 12.8
- TorchVision 0.23.0 with CUDA 12.8
- NumPy 2.0.2
- Pillow 11.3.0

## 1. Create an isolated environment

Using Conda:

```bash
conda create -n paum-rockfsl python=3.9 -y
conda activate paum-rockfsl
```

Alternatively, using `venv` on Windows:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install dependencies

For an NVIDIA GPU with CUDA 12.8-compatible drivers:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For CPU-only execution:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The `--device cpu` option can be used for functional checks, but formal training is intended for a CUDA-capable GPU.

## 3. Verify the installation

```bash
python -c "import torch, torchvision; import paum_rock_fsl; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Run all commands from the repository root so that `paum_rock_fsl` can be imported without an additional installation step.
