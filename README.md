# Interpretable Linear Directions in JetClass Transformers

Finding linear directions that correspond to interpretable physics concepts in a transformer trained on [JetClass](https://github.com/jet-universe/particle_transformer)
jet-tagging data.

The model is a plain transformer (`jetdirections.model.transformer.JetTagTransformer`) trained to classify jets into 10 physics classes from per-particle features.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
uv sync
```

This creates a virtual environment and installs the `jetdirections` package in editable
mode (along with the dev tools).

## Data

JetClass ROOT files are not tracked in git. The code expects them split into
three flat directories:

```
data/
├── train/   # *.root
├── val/     # *.root
└── test/    # *.root
```

To download the raw JetClass archives, install [`aria2c`](https://aria2.github.io/)
and run the downloader on Linux or macOS:

```bash
bash scripts/download_data.sh
```

On Windows, run it through WSL or Git Bash with `aria2c` installed. The script
stores the archives in `data/`.

Then run the cross-platform Python extraction helper to produce the layout
above:

```bash
uv run python scripts/extract_data.py
```

You can remove the downloaded archives after successful extraction:

```bash
uv run python scripts/extract_data.py --delete-tars
```

Hyperparameters and data paths are configured in
[`src/jetdirections/config.py`](src/jetdirections/config.py).

## Model checkpoint

The workshop experiments use a pretrained checkpoint. Download it into the
expected location with:

```bash
bash scripts/download_model.sh
```

This saves the verified model as `checkpoints/jetclass_transformer.pt`.

## Usage

Train (optional; the workshop notebooks use the supplied checkpoint):

```bash
uv run jetdirections-train
# equivalently: uv run python -m jetdirections.training.train
```

Training settings are configured in `src/jetdirections/config.py`.

Monitor with TensorBoard:

```bash
uv run tensorboard --logdir runs
```

Evaluate a checkpoint on the test set:

```bash
uv run jetdirections-eval --checkpoint checkpoints/jetclass_transformer.pt
```

## Layout

```
src/jetdirections/
├── config.py
│   └── Shared model, training, and data-path settings
├── data/
│   ├── dataset.py
│   │   └── Reads JetClass ROOT files one file at a time
│   ├── dataloader.py
│   │   └── Buffers files, mixes classes, and yields training batches
│   ├── sampling.py
│   │   └── Selects class-balanced and concept-ranked batches
│   └── vectorize.py
│       └── Converts ROOT branches into padded, normalized tensors
├── model/
│   └── transformer.py
│       └── Defines the model architecture
├── interp/
│   ├── activations.py
│   │   └── Extracts model activations
│   ├── contrast.py
│   │   └── Constructs matched concept contrasts
│   ├── probes.py
│   │   └── Fits linear probes
│   └── steering.py
│       └── Extracts and applies steering directions
└── training/
    ├── engine.py
    │   └── Training and evaluation loops
    ├── evaluate.py
    │   └── Evaluates a checkpoint on the test set
    ├── metrics.py
    │   └── Computes classification metrics
    ├── plot.py
    │   └── Evaluates checkpoint convergence
    ├── train.py
    │   └── Trains the model
    └── utils.py
        └── Checkpoint and training utilities
```
