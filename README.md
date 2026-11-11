
# COMP8240 TabM Replication and Extension

This repository contains the code, experiment records and results for my COMP8240 Applications of Data Science project reproducing and extending:

**Gorishniy, Y., Kotelnikov, A. & Babenko, A. (2025), _TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling_.**

The project reproduces the California Housing MLP vs TabM experiment and extends the comparison to:

1. Power Consumption of Tetouan City, and
2. controlled synthetic regression datasets with varying training size and target noise.

## Original implementation

The experiments use the authors' official TabM repository:

https://github.com/yandex-research/tabm

The original repository was fixed to commit:

`28e47ae301c92ec37787dde1ce923a0793f405b4`

The main experimental environment used in Google Colab was:

- Python 3.10.12
- PyTorch 2.0.1
- CUDA 11.8
- NVIDIA Tesla T4

The official TabM training pipeline was used rather than reimplementing the models.

The main model implementation used for both MLP and TabM is:

`bin/model.py`

## Repository structure

- `scripts/prepare_tetouan.py`  
  Downloads the Tetouan dataset from UCI, constructs cyclical time features and creates the chronological train/validation/test split.

- `scripts/generate_synthetic_regression.py`  
  Reproducibly generates the nine synthetic regression conditions.

- `scripts/run_experiments.py`  
  Provides an executable wrapper for running the California, Tetouan and synthetic MLP/TabM experiments using the official TabM training pipeline.

- `data/README.md`  
  Documents the original, external and constructed datasets. Large generated arrays are intentionally excluded from Git.

- `docs/tetouan_experiment_notes.md`  
  Documents the final Tetouan experiment design and results.

- `docs/synthetic_experiment_notes.md`  
  Documents the final synthetic experiment design and results.

- `docs/reproduction_commands.md`  
  Records the environment and reproduction workflow.

- `results/`  
  Contains saved experiment summaries and the supplementary LLM-as-judge evaluation materials.

## Data preparation

### Tetouan City Power Consumption

Run:

```bash
pip install numpy pandas ucimlrepo
python scripts/prepare_tetouan.py
```

The script downloads UCI dataset ID 849 and creates:

- 33,546 training observations
- 8,386 validation observations
- 10,484 test observations
- 11 numerical predictors
- Zone 1 Power Consumption as the target

A chronological 64% / 16% / 20% split is used.

### Synthetic regression

Run:

```bash
python scripts/generate_synthetic_regression.py
```

The script generates nine conditions from:

- training sizes: 500, 2,000 and 8,000
- target noise: 0%, 10% and 30% of clean signal standard deviation
- 10 numerical predictors
- construction seed: 42

Generated NumPy arrays are stored under `data/processed/` and are excluded from version control.

## Running the experiments

The repository includes an executable wrapper around the official TabM training pipeline:

`scripts/run_experiments.py`

The runner expects the official Yandex Research TabM repository and its corresponding environment.

### California Housing replication

```bash
python scripts/run_experiments.py california \
  --tabm-paper-dir /content/tabm/paper \
  --micromamba-prefix /content/tabm-env
```

This evaluates the released California Housing MLP and TabM configurations across five seeds.

### Tetouan City Power Consumption

```bash
python scripts/run_experiments.py tetouan \
  --tabm-paper-dir /content/tabm/paper \
  --micromamba-prefix /content/tabm-env
```

This first recreates the processed Tetouan dataset using:

`scripts/prepare_tetouan.py`

and then evaluates MLP and TabM across three seeds.

### Synthetic regression

```bash
python scripts/run_experiments.py synthetic \
  --tabm-paper-dir /content/tabm/paper \
  --micromamba-prefix /content/tabm-env
```

This recreates all nine synthetic conditions using:

`scripts/generate_synthetic_regression.py`

and evaluates MLP and TabM using three seeds per condition.

### Run all experiments

```bash
python scripts/run_experiments.py all \
  --tabm-paper-dir /content/tabm/paper \
  --micromamba-prefix /content/tabm-env
```

The experiment runner uses the authors' official `bin/model.py` implementation.

For Tetouan and the synthetic datasets, the released California Housing configurations are transferred while changing the dataset path and random seed.

No dataset-specific hyperparameter optimisation is performed for these extension experiments.

Further details are provided in:

`docs/reproduction_commands.md`

## Main results

### California Housing replication

Published results:

- MLP: 0.4948 ± 0.0058 RMSE
- TabM: 0.4414 ± 0.0012 RMSE

Replication across five seeds:

- MLP: 0.4939 ± 0.0055 RMSE
- TabM: 0.4416 ± 0.0021 RMSE

The replicated model ordering and mean performance closely match the published result.

The saved summary is:

`results/california_5seed_results.json`

### Tetouan City Power Consumption

Across three seeds:

- MLP: 4752.23 ± 286.36 RMSE
- TabM: 4840.42 ± 183.92 RMSE

TabM therefore did not consistently outperform the MLP under the transferred California configurations.

The direction varied across seeds: MLP performed better for seeds 0 and 1, while TabM performed better for seed 2.

### Synthetic regression

TabM obtained lower mean RMSE in all nine synthetic conditions.

Its relative RMSE improvement decreased as target noise increased:

- 0% noise: approximately 29–31%
- 10% noise: approximately 14–25%
- 30% noise: approximately 3–9%

This pattern was observed across the three tested training sizes.

### LLM-as-judge

A supplementary blinded pairwise evaluation used 12 California Housing cases where the seed-0 MLP and TabM predictions differed strongly.

The true target and model identities were hidden from the judge.

Each case was evaluated twice:

1. original A/B ordering
2. reversed A/B ordering

Final results:

- Original-order accuracy: 66.7%
- Reversed-order accuracy: 58.3%
- Position consistency: 75.0%
- Position-sensitive cases: 3/12

The LLM evaluation is treated only as supplementary qualitative evidence.

RMSE remains the primary evaluation metric.

## Reproducibility notes

Large raw and processed datasets are not committed to the repository.

The supplied preparation scripts recreate the new datasets.

The model experiments depend on the official TabM codebase and released configurations at the fixed commit specified above.

Dataset-specific hyperparameter optimisation was not performed for Tetouan or the synthetic datasets.

California Housing configurations were deliberately transferred to the new-data experiments in order to test how the released configuration behaves under different datasets and controlled data conditions.

The project does not claim that these transferred configurations are optimal for Tetouan or the synthetic datasets.

## Author

Thi Kieu Ngan Nguyen  
Macquarie University  
COMP8240 — Applications of Data Science
