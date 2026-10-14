# Data

Large/raw datasets are not stored directly in this repository.

This file records the data sources and the intended reproducible preparation process.

## Original replication

### California Housing
- Used for the original TabM replication.
- Source: authors' released TabM experimental data.
- Task: regression.
- Metric: RMSE.

## New existing data

### Power Consumption of Tetouan City
- Source: UCI Machine Learning Repository.
- DOI: 10.24432/C5B034
- Task: regression.
- The project will select one electricity-consumption zone as the target.
- A chronological train/validation/test split will be used because the data are time ordered.

## Constructed data

### Synthetic tabular regression
- Generated using a reproducible Python script.
- The underlying generating process will be fixed.
- Training-set size and target noise will be varied systematically.
- MLP and TabM will be compared using RMSE and training time.
