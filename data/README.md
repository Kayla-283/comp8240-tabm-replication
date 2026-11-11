# Project Data

This repository does not store the large generated NumPy arrays used by the
experiments. Instead, the supplied preparation scripts reproduce the new data.

## Original replication data

The California Housing replication uses the dataset and prepared benchmark
format released with the official TabM implementation.

Official TabM repository:
https://github.com/yandex-research/tabm

The project used commit:

`28e47ae301c92ec37787dde1ce923a0793f405b4`

## Tetouan City Power Consumption

Source: UCI Machine Learning Repository, dataset ID 849.

The downloaded dataset contains 52,416 observations recorded at ten-minute
intervals during 2017.

For this project:

- Zone 1 Power Consumption is the regression target.
- Zone 2 and Zone 3 are excluded as predictors.
- DateTime is converted into daily, weekly and yearly sine/cosine features.
- Five environmental variables are retained.
- The final model input contains 11 numerical predictors.
- A chronological 64% / 16% / 20% train-validation-test split is used.

Final split sizes:

- Training: 33,546
- Validation: 8,386
- Test: 10,484

Recreate the processed data with:

python scripts/prepare_tetouan.py

## Synthetic regression data

The constructed synthetic experiment contains ten standard-normal predictors
and a fixed nonlinear regression function.

Training sizes:

- 500
- 2,000
- 8,000

Target-noise levels:

- 0%
- 10%
- 30% of the clean target signal standard deviation

This produces nine experimental conditions.

Validation size is 2,000 and test size is 4,000 for every condition.

Recreate the datasets with:

python scripts/generate_synthetic_regression.py

Generated arrays are written under `data/processed/` and are intentionally
excluded from Git using `.gitignore`.
