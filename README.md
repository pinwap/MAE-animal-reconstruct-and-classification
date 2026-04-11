# MAE-animal-reconstruct-and-classification

Kaggle-first implementation for Animal Image Reconstruction and Classification using a Masked Autoencoder.

This project supports two execution styles:
- Notebook-first on Kaggle: [start_implementation.ipynb](start_implementation.ipynb)
- Script/CLI on Kaggle or local: [main.py](main.py)

## Start Here

- Open [start_implementation.ipynb](start_implementation.ipynb) for the main Kaggle workflow.
- Use `/kaggle/input/animals10/raw-img` as the default dataset root.
- Keep frequently edited training logic in the notebook and reusable code in the `data/`, `models/`, `training/`, and `utils/` modules.

## Core Modules

- [mae_core.py](mae_core.py) for MAE loading and reconstruction helpers.
- [data/animals10.py](data/animals10.py) for dataset loading and splits.
- [models/unet.py](models/unet.py) for the U-Net baseline.
- [training/](training) for training loops, evaluation, and checkpoint helpers.

## What To Zip For Kaggle

When you upload your own code as a Kaggle Dataset, zip only project code (not environment and git metadata):

- Include:
	- [main.py](main.py)
	- [mae_core.py](mae_core.py)
	- [start_implementation.ipynb](start_implementation.ipynb)
	- [data/](data)
	- [models/](models)
	- [training/](training)
	- [utils/](utils)
	- [pyproject.toml](pyproject.toml)
	- [README.md](README.md)
- Exclude:
	- `.venv/`
	- `.git/`
	- `__pycache__/`
	- large outputs/checkpoints you do not want to version

PowerShell example (run in project root):

```powershell
Compress-Archive -Path main.py,mae_core.py,start_implementation.ipynb,data,models,training,utils,pyproject.toml,README.md -DestinationPath kaggle_code_bundle.zip -Force
```

## Kaggle Run Steps (Recommended)

1. Create a new Kaggle Notebook.
2. Add dataset `animals10` (so images are at `/kaggle/input/animals10/raw-img`).
3. Add your zipped code dataset (`kaggle_code_bundle.zip`) or upload files directly.
4. In a first setup cell, unzip code to writable working directory:

```python
!unzip -q /kaggle/input/<your-code-dataset>/kaggle_code_bundle.zip -d /kaggle/working/project
%cd /kaggle/working/project
```

5. Install dependencies (if Kaggle image does not already include them):

```python
!pip install -q torch torchvision transformers matplotlib
```

6. Run notebook workflow in [start_implementation.ipynb](start_implementation.ipynb) or run CLI commands below.

## CLI Usage

Default dataset root is already Kaggle path (`/kaggle/input/animals10/raw-img`).

Quick check:

```bash
python main.py --step setup-check
```

Train MAE:

```bash
python main.py --step train-mae --epochs-mae 10 --batch-size 32 --checkpoint-every 2
```

Train U-Net baseline:

```bash
python main.py --step train-unet --epochs-unet 10 --batch-size 32 --checkpoint-every 2
```

Train classifier (with MAE initialization):

```bash
python main.py --step train-cls --epochs-cls 10 --mae-checkpoint /kaggle/working/checkpoints/mae/best.pt
```

Run final comparison + metrics + sample PNG:

```bash
python main.py --step eval-compare
```

Run all steps end-to-end:

```bash
python main.py --step all --epochs 5 --checkpoint-every 1
```

## Output Locations On Kaggle

- Checkpoints: `/kaggle/working/checkpoints/...`
- Metrics JSON: `/kaggle/working/results/...`
- Comparison image: `/kaggle/working/results/comparison/sample.png`