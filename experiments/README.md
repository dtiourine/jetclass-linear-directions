# Clean experiments

This directory contains thin, paper-aligned notebooks that reproduce the reported interpretability results. Scientific definitions and experiment orchestration remain visible in each notebook; reusable tooling under `src/jetdirections/interp/` is concept agnostic.

## Physical-concept experiments

- `01_physical_concept_probes.ipynb` fits held-out linear probes for jet mass, particle multiplicity, jet charge, and track displacement at transformer blocks 0–5.
- `02_physical_concept_directions.ipynb` extracts and saves the four matched, per-block concept directions.
- `03_physical_concept_steering.ipynb` applies those directions, produces decision-migration plots, and displays appendix tables of predicted-class fractions.
- `04_physical_concept_downstream_probes.ipynb` reuses the block 1–4 steering from `03` and reads the block-5 probes fit in `01` (loaded, not refit) to test whether steering shifts the model's downstream linear representation of each concept, evaluated on a class-balanced mixed batch.

From the repository root, start Jupyter in the project environment:

```bash
uv run jupyter lab
```

Run the notebook from top to bottom. It reads `checkpoints/jetclass_transformer.pt` and validation ROOT files from `data/val/`, then writes machine-readable metrics and figures to `experiments/results/physical_concept_probes/`.
