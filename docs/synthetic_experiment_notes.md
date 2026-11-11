# Synthetic Regression Experiment Notes

## Purpose

The synthetic experiment tests the MLP–TabM comparison under controlled
changes in training-set size and target noise while holding the generating
relationship fixed.

## Generating process

Ten predictors are sampled independently from a standard normal distribution.

The clean target is:

y = 3*x0 - 2*x1 + 1.5*x2*x3 + 2*sin(x4) + 0.8*x5^2 - 1.2*x6

Predictors x7, x8 and x9 do not contribute to the target.

Construction seed: 42.

## Conditions

Training sizes:

- 500
- 2,000
- 8,000

Target-noise levels:

- 0%
- 10%
- 30% of clean signal standard deviation

This gives nine total conditions.

Validation size: 2,000.

Test size: 4,000.

Gaussian target noise is applied to the training, validation and test targets
for each noisy condition.

## Experimental design

California Housing MLP and TabM configurations are transferred without
synthetic-specific hyperparameter optimisation.

Each condition is evaluated using three model seeds.

## Final results

| Train size | Noise | MLP RMSE | TabM RMSE | TabM improvement |
|---:|---:|---:|---:|---:|
| 500 | 0% | 1.0176 | 0.6973 | 31.47% |
| 500 | 10% | 1.1123 | 0.8387 | 24.60% |
| 500 | 30% | 1.8534 | 1.6879 | 8.93% |
| 2,000 | 0% | 0.5832 | 0.4136 | 29.08% |
| 2,000 | 10% | 0.7742 | 0.6352 | 17.96% |
| 2,000 | 30% | 1.5612 | 1.4762 | 5.45% |
| 8,000 | 0% | 0.3401 | 0.2385 | 29.88% |
| 8,000 | 10% | 0.5955 | 0.5144 | 13.61% |
| 8,000 | 30% | 1.4792 | 1.4275 | 3.50% |

TabM achieves lower mean RMSE in all nine conditions.

Its relative advantage decreases consistently as target noise increases. The
strongest gains occur in the clean-target conditions, while the difference
becomes much smaller at 30% target noise.
