# Synthetic Regression Experiment Notes

## Construction
- Synthetic tabular regression data were generated using a reproducible Python script.
- 10 numerical predictors were generated from a fixed random process.
- A fixed nonlinear generating function defines the clean continuous target.
- The underlying relationship is kept constant across experimental conditions.
- Random seed for data construction: 42.

## Controlled conditions
Training-set size:
- 500
- 2,000
- 8,000

Target noise:
- 0%
- 10%
- 30% of the clean signal standard deviation

This creates nine sample-size × noise conditions from the same underlying generating process.

## Pilot condition tested for Project Update
- Training observations: 500
- Validation observations: 2,000
- Test observations: 4,000
- Target noise: 30%
- Model seed: 0
- Metric: RMSE

## Preliminary result
- MLP test RMSE: 1.8543
- MLP test R²: 0.8463
- MLP training time: 2.48 seconds
- TabM test RMSE: 1.6765
- TabM test R²: 0.8743
- TabM training time: 11.46 seconds
- TabM achieved 9.59% lower RMSE than MLP.

## Interpretation
The preliminary result is consistent with the possibility that TabM's
parameter-efficient ensemble may help under a small and noisy data condition.

However, this is only one condition and one model seed. The remaining
sample-size/noise conditions and repeated seeds are required before making
a final claim about the relationship between noise, data availability and
TabM's advantage.

## Important Project Update comparison
The new-data experiments currently show different behaviour:

- Tetouan real dataset: TabM RMSE was 6.76% worse than MLP.
- Synthetic n=500, 30% noise: TabM RMSE was 9.59% better than MLP.

This contrast motivates the continuing research question: under what data
conditions does parameter-efficient ensembling provide a meaningful benefit?
