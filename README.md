# Tiny-Defect-VQA

Implementation of **Tiny Defect-VQA: Mask-Grounded Edge Visual Question Answering for Industrial Surface Inspection**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate synthetic defect rectangles with presence, left/right/absent and area-bin questions. Three questions from one image share a group; token 0 is padding.

```bash
tiny-defect-vqa demo --output data/generated.npz --count 96 --config config.json --seed 13
tiny-defect-vqa prepare --input data/generated.npz --output data/prepared
```

`x[N,3,H,W]`, integer token IDs `question[N,L]`, binary defect `mask[N,1,H,W]`, and integer answer `y[N]` describe grounded examples. Export questions, answers, and masks from a declared inspection/VQA annotation protocol; freeze the vocabulary on training questions and reserve token 0 for padding. Set `vocabulary` and `classes` accordingly.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
tiny-defect-vqa train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

## Evaluation

```bash
tiny-defect-vqa evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

Optional confidence calibration uses its independent split:

```bash
tiny-defect-vqa calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
tiny-defect-vqa evaluate --input data/prepared/test.npz --model runs/full/calibrated.pt --output runs/full/calibrated_evaluation
```

## Ablations

Variants: `full`, `baseline`, `no_mask`, `no_gate`, `no_localization`. Reuse the prepared data and training seed:

```bash
tiny-defect-vqa train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
tiny-defect-vqa evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
tiny-defect-vqa plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
