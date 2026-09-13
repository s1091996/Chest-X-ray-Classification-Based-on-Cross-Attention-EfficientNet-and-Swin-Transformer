# Chest X-ray Multi-label Classification

> Multi-label chest X-ray classification on the NIH ChestX-ray14 dataset, with an EfficientNet baseline and an experimental EfficientNet--Swin Transformer cross-attention fusion architecture.

## Overview

This project explores automatic detection of thoracic findings from chest X-ray images. It treats the task as **multi-label classification**: one image may contain more than one finding. The codebase includes a complete training and evaluation workflow, data augmentation, per-label threshold selection, mixed-precision training, gradient accumulation, and early stopping.

The project is intended for research and portfolio purposes only. It is **not a medical device** and must not be used for clinical diagnosis or treatment decisions.

## Highlights

- **15-label prediction**: 14 thoracic findings plus `No Finding`.
- **NIH ChestX-ray14 support** with patient-level CSV splits.
- **EfficientNet-B5 baseline** used by the current training entry point.
- **Experimental dual-stream architecture** that combines EfficientNet and Swin Transformer features through cross-attention, residual connections, positional embeddings, and a feed-forward block.
- **Evaluation per label**: accuracy, F1, precision, recall, sensitivity, specificity, ROC-AUC, and an F1-optimized decision threshold.
- **Training utilities**: AMP, gradient accumulation, checkpointing, and early stopping based on validation macro F1.

## Project Structure

```text
.
├── config.py                  # Hyperparameters, labels, and local data/output paths
├── main.py                    # Current baseline training and test evaluation entry point
├── model.py                   # Experimental EfficientNet + Swin cross-attention model
├── cross_attention_model.py   # Earlier experimental cross-attention variant
├── model_manager.py           # Training loop, AMP, checkpoints, early stopping
├── dataset.py                 # Dataset class and image transforms
├── dataloader.py              # Train/validation/test DataLoaders
├── evaluate.py                # Metrics and per-label threshold search
├── print_config.py            # Configuration display helper
└── requirements.txt           # Python dependencies
```

## Model Status

| Component | Notes |
| --- | --- |
| EfficientNet-B5 baseline | `main.py` currently creates this model with `timm.create_model(...)`. |
| Cross-attention fusion model | Implemented in `model.py`, but is not yet wired into `main.py`'s training flow. |

Keeping this distinction visible makes the repository reproducible and accurately represents the current implementation.

## Dataset

The code expects the [NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) and three split CSV files:

```text
<dataset root>/
├── images_001/images/
├── ...
├── images_012/images/
└── <split folder>/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Each CSV must contain an `Image` column and these 15 binary label columns:

```text
Atelectasis, Hernia, Cardiomegaly, Infiltration, Consolidation,
Mass, Edema, Nodule, Effusion, Pleural_Thickening, Emphysema,
Pneumonia, Fibrosis, Pneumothorax, No Finding
```

Do not commit the NIH images or CSV splits if their licence, distribution terms, or patient-data policy does not allow redistribution.

## Setup

Tested configuration: Python 3.10+ with a CUDA-enabled PyTorch installation recommended for training.

```bash
git clone <your-repository-url>
cd Chest-X-ray-Classification-Based-on-Cross-Attention-EfficientNet-and-Swin-Transformer
python -m venv .venv
```

Activate the environment, install the appropriate PyTorch build for your system from [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
pip install -r requirements.txt
```

Before running, edit the path variables in `config.py` for your machine:

```python
MODEL_FILE_PATH = "<path for checkpoints and metrics>"
DATASET_FILE_PATH = "<path containing train.csv, val.csv, and test.csv>"
IMAGE_ROOT = "<path to NIH ChestXray14 images>"
```

## Train and Evaluate

```bash
python main.py
```

The workflow trains the EfficientNet baseline, selects validation thresholds that maximize per-label F1, loads the best validation-F1 checkpoint, then writes evaluation results to the configured output directory.

## Metrics

For each label, the evaluation script reports:

- Accuracy
- F1 score
- Precision and recall
- Sensitivity and specificity
- ROC-AUC
- The decision threshold selected from validation predictions

Macro averages are also calculated across labels. Add a results table, training curves, confusion matrices, or ROC plots here once you have exportable experiment outputs; these are especially valuable for a portfolio repository.

## Reproducibility Notes

- Set fixed random seeds if you need strictly repeatable experiments.
- Keep training configuration and split definitions versioned alongside experiment results.
- Record GPU model, PyTorch/CUDA versions, epoch count, and the selected checkpoint when publishing metrics.

## Roadmap

- [ ] Connect `DualStreamModel` to the main training entry point.
- [ ] Add auxiliary-loss handling for its multiple training outputs.
- [ ] Add a configuration template or command-line arguments to remove machine-specific paths.
- [ ] Publish a reproducible experiment table and visualizations.

## License

No license is currently included. Add a license before inviting others to reuse or distribute the code (MIT is a common choice for portfolio projects).
