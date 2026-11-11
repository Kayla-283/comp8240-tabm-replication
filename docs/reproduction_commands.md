# Reproduction Commands

This document records the environment and commands used to reproduce the COMP8240 TabM replication and extension experiments.

## 1. Official TabM implementation

The experiments use the authors' official repository:

https://github.com/yandex-research/tabm

The repository was fixed to commit:

`28e47ae301c92ec37787dde1ce923a0793f405b4`

The MLP and TabM models are trained using the authors' original:

`paper/bin/model.py`

implementation.

## 2. Experimental environment

The main experiments were run in Google Colab using:

- Python 3.10.12
- PyTorch 2.0.1
- CUDA 11.8
- NVIDIA Tesla T4

The TabM environment used in Colab was located at:

`/content/tabm-env`

The official TabM repository was located at:

`/content/tabm`

## 3. Experiment runner

This repository provides:

`scripts/run_experiments.py`

The script wraps the official TabM training pipeline and provides reproducible commands for the experiments used in the report.

### California Housing replication

```bash
python scripts/run_experiments.py california \
  --tabm-paper-dir /content/tabm/paper \
  --micromamba-prefix /content/tabm-env
```

This evaluates MLP and TabM across five seeds using the released California Housing configurations.

### Tetouan City Power Consumption

```bash
python scripts/run_experiments.py tetouan \
  --tabm-paper-dir /content/tabm/paper \
  --micromamba-prefix /content/tabm-env
```

The runner first executes:

`scripts/prepare_tetouan.py`

to recreate the processed Tetouan dataset and then evaluates MLP and TabM across three seeds.

The California Housing configurations are transferred without Tetouan-specific hyperparameter optimisation.

### Synthetic regression

```bash
python scripts/run_experiments.py synthetic \
  --tabm-paper-dir /content/tabm/paper \
  --micromamba-prefix /content/tabm-env
```

The runner first executes:

`scripts/generate_synthetic_regression.py`

to recreate the nine synthetic dataset conditions and then evaluates MLP and TabM across three seeds for each condition.

The California Housing configurations are transferred without synthetic-specific hyperparameter optimisation.

### Run all experiments

```bash
python scripts/run_experiments.py all \
  --tabm-paper-dir /content/tabm/paper \
  --micromamba-prefix /content/tabm-env
```

## 4. Dataset preparation only

Tetouan:

```bash
python scripts/prepare_tetouan.py
```

Synthetic regression:

```bash
python scripts/generate_synthetic_regression.py
```

Large generated arrays are excluded from Git and can be recreated using these scripts.

## 5. Saved results

Final experiment summaries are stored under:

`results/`

This includes:

- California Housing five-seed replication results
- Tetouan three-seed results
- synthetic three-seed experiment results
- supplementary LLM-as-judge prompts and evaluation results

## 6. Reproducibility scope

The project uses the authors' official model implementation rather than copying or reimplementing TabM.

For Tetouan and synthetic experiments, the released California Housing configurations are deliberately transferred to the new datasets.

The project does not claim that these configurations are optimal for the new datasets. Their purpose is to evaluate how the released configuration transfers under different real-world and controlled synthetic conditions.
