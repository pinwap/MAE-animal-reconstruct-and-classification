# MAE-animal-reconstruct-and-classification

Kaggle-first, notebook-only pipeline for Animals-10:
- MAE reconstruction pretraining
- U-Net reconstruction baseline
- MAE-encoder classification fine-tuning

Execution entrypoint:
- start_implementation.ipynb

## File Responsibilities

| File | Role | Edit Frequency |
|---|---|---|
| start_implementation.ipynb | Main orchestrator on Kaggle, config, step-by-step training/eval flow | High |
| data/animals10.py | Data discovery, split, transforms, dataloaders | Low |
| models/unet.py | U-Net architecture definition | Low |
| training/mae_trainer.py | MAE load/process/reconstruct + MAE train/eval loop | Low |
| training/unet.py | Patch masking + U-Net train/eval loop | Low |
| training/classification.py | MAE->classifier weight transfer + classifier train/eval | Low |
| training/evaluation.py | MAE vs U-Net comparison metrics + classifier metric wrapper | Low |
| utils/common.py | Shared seed/device/mixed-precision/checkpoint/json + visualization helpers | Low |

## Call Graph

```mermaid
flowchart TD
    N[start_implementation.ipynb] --> D[data/animals10.py]
    N --> M[training/mae_trainer.py]
    N --> U[training/unet.py]
    N --> C[training/classification.py]
    N --> E[training/evaluation.py]
    N --> G[utils/common.py]

    U --> UM[models/unet.py]
    E --> U
    E --> M
    E --> C
    E --> G

    M --> G
    U --> G
    C --> G
```

## Kaggle Packaging

Zip only code files (do not include .venv/.git/cache/output):

```powershell
Compress-Archive -Path start_implementation.ipynb,data,models,training,utils,pyproject.toml,README.md -DestinationPath kaggle_code_bundle.zip -Force
```

## Kaggle Run Steps

1. Create a new Kaggle Notebook.
2. Add dataset `alessiocorrado99/animals10` so images are at:
    `/kaggle/input/datasets/alessiocorrado99/animals10/raw-img`
3. Add your code zip as a Kaggle Dataset.
4. In notebook Section 0, set:
    - `USE_ZIPPED_PROJECT_CODE = True` (if you uploaded a zip dataset)
    - `CODE_ZIP_PATH` to your actual zip path
5. Install dependencies if needed:
```python
!pip install -q torch torchvision transformers matplotlib
```
6. Run notebook from Section 0 to the end.

## Classifier Training Mode Switch

You can choose how classification fine-tuning works from Cell 1 in notebook:

- `CLS_TRAIN_MODE = "end_to_end"`:
    train all classifier parameters.
- `CLS_TRAIN_MODE = "partial"`:
    freeze almost everything and train only:
    - classifier head
    - last `UNFREEZE_LAST_BLOCKS` ViT blocks
    - optional final ViT layer norm (`UNFREEZE_VIT_LAYERNORM`)

This mode is applied in notebook Cell 6 and optimizer in Cell 7 automatically
uses only trainable parameters.

## Outputs On Kaggle

- Checkpoints: /kaggle/working/checkpoints/...
- Metrics JSON: /kaggle/working/results/...
- Comparison image: /kaggle/working/results/comparison/sample_comparison.png

## About Editing .py Files

Default workflow:
- Usually you do not need to edit .py files every run.
- You call functions/classes from notebook cells.
- Tune hyperparameters and experiment flow in start_implementation.ipynb.

When should you edit .py files:
- New model architecture
- New augmentation or split policy
- New metric, checkpoint format, or visualization behavior

In short: yes, you can call these .py modules from ipynb directly as reusable functions/classes.
