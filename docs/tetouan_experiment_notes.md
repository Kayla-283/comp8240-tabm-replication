# Tetouan Experiment Notes

## Dataset

Power Consumption of Tetouan City, UCI Machine Learning Repository.

Actual downloaded observations: 52,416.

Zone 1 Power Consumption is used as the regression target. Zone 2 and Zone 3
are excluded from the predictors.

DateTime is transformed into cyclical representations of:

- time of day;
- day of week; and
- position within the year.

Together with the five environmental measurements, the processed dataset has
11 numerical predictors.

## Split

A chronological split is used:

- Train: 33,546 observations (64%)
- Validation: 8,386 observations (16%)
- Test: 10,484 observations (20%)

## Experimental design

The released California Housing MLP and TabM configurations are transferred to
Tetouan without Tetouan-specific hyperparameter optimisation.

Three model seeds are evaluated.

## Final results

| Model | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| MLP | 4424.00 | 4881.65 | 4951.03 | 4752.23 ± 286.36 |
| TabM | 4723.17 | 5052.40 | 4745.69 | 4840.42 ± 183.92 |

TabM's mean RMSE is approximately 1.86% higher than MLP.

The direction is not consistent across seeds: MLP performs better for seeds 0
and 1, while TabM performs better for seed 2. The result is therefore treated
as evidence that the strong California Housing advantage does not
automatically transfer to this dataset under the transferred configurations.
