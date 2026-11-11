# Reproduction Commands

This document records the environment, data-preparation and model-running workflow used for the COMP8240 TabM replication and extension project.

## 1. Official TabM implementation

Official repository:

https://github.com/yandex-research/tabm

Fixed commit:

`28e47ae301c92ec37787dde1ce923a0793f405b4`

The main model implementation used for both MLP and TabM is:

`bin/model.py`

## 2. Experimental environment

Experiments were run in Google Colab using:

- Python 3.10.12
- PyTorch 2.0.1
- CUDA 11.8
- NVIDIA Tesla T4 GPU

An isolated environment was used because the default Colab software stack was newer than the versions expected by the released repository.

## 3. California Housing replication

California Housing uses the original prepared benchmark data and released model configurations from the official TabM repository.

The objective was to reproduce the reported MLP-versus-TabM comparison without repeating the original hyperparameter search.

The official `bin/model.py` training pipeline was used with the released California Housing configurations for MLP and TabM across five seeds.

Final replication results:

- MLP: 0.4939 ± 0.0055 RMSE
- TabM: 0.4416 ± 0.0021 RMSE

Published results:

- MLP: 0.4948 ± 0.0058 RMSE
- TabM: 0.4414 ± 0.0012 RMSE

Saved summary:

`results/california_5seed_results.json`

## 4. Tetouan City Power Consumption

Prepare the dataset with:

`python scripts/prepare_tetouan.py`

The script:

- downloads the UCI Tetouan dataset;
- selects Zone 1 Power Consumption as the regression target;
- excludes Zone 2 and Zone 3 from predictors;
- converts DateTime into daily, weekly and yearly cyclical features;
- produces 11 numerical predictors;
- creates a chronological 64% / 16% / 20% split.

Final split sizes:

- Train: 33,546
- Validation: 8,386
- Test: 10,484

The released California Housing MLP and TabM configurations were transferred to Tetouan without Tetouan-specific hyperparameter optimisation.

Three model seeds were evaluated.

Final mean RMSE:

- MLP: 4752.23 ± 286.36
- TabM: 4840.42 ± 183.92

Relevant files:

- `scripts/prepare_tetouan.py`
- `docs/tetouan_experiment_notes.md`
- `results/`

## 5. Synthetic regression experiment

Generate the datasets with:

`python scripts/generate_synthetic_regression.py`

Training sizes:

- 500
- 2,000
- 8,000

Target noise:

- 0%
- 10%
- 30% of clean signal standard deviation

Clean regression function:

`y = 3*x0 - 2*x1 + 1.5*x2*x3 + 2*sin(x4) + 0.8*x5^2 - 1.2*x6`

Construction seed:

`42`

Validation size:

`2,000`

Test size:

`4,000`

California Housing MLP and TabM configurations were transferred without synthetic-specific hyperparameter optimisation.

Each condition was evaluated using three model seeds.

Relevant files:

- `scripts/generate_synthetic_regression.py`
- `docs/synthetic_experiment_notes.md`
- `results/`

## 6. LLM-as-judge evaluation

The LLM-as-judge analysis is supplementary and does not replace RMSE.

Twelve California Housing test cases were selected using the largest absolute disagreement between the seed-0 MLP and TabM predictions.

The true target and model identities were hidden from the judge.

Each case was evaluated twice:

1. original A/B ordering;
2. reversed A/B ordering.

Final summary:

- Original-order accuracy: 66.7%
- Reversed-order accuracy: 58.3%
- Position consistency: 75.0%
- Position-sensitive cases: 3/12

The corresponding prompts and results are stored in `results/`.

## 7. Reproducibility notes

Large generated NumPy arrays are excluded from Git and can be recreated using the supplied preparation scripts.

The transferred configurations are not claimed to be optimal for Tetouan or the synthetic datasets. They are used to test how the released California Housing configurations transfer to new data under a controlled setup.
