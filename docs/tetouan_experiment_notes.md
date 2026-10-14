# Tetouan Experiment Notes

## Dataset
- Dataset: Power Consumption of Tetouan City
- Source: UCI Machine Learning Repository
- Proposed observation count: 52,417
- Actual observations loaded through UCI: 52,416
- Actual date range: 2017-01-01 00:00 to 2017-12-30 23:50
- Observations occur every 10 minutes.
- No missing values were found.
- No duplicate timestamps were found.

## Prediction task
- Task type: regression
- Selected target: Zone 1 Power Consumption
- Zone 2 and Zone 3 Power Consumption are excluded from the predictors.
- Evaluation metric: RMSE.

## Original predictors
The UCI dataset provides:
- DateTime
- Temperature
- Humidity
- Wind Speed
- general diffuse flows
- diffuse flows

## DateTime preprocessing
Raw DateTime is not supplied directly to the neural network.

It is transformed into:
- time_sin
- time_cos
- weekday_sin
- weekday_cos
- year_sin
- year_cos

Together with the five environmental variables, this produces 11 numerical model predictors.

This cyclical encoding is a preprocessing decision made for the new-data extension and is not part of the original California Housing experiment.

## Data split
A chronological split is used instead of a random split to respect the time-ordered structure.

- Training: 33,546 rows (64%)
  - 2017-01-01 00:00 to 2017-08-21 22:50
- Validation: 8,386 rows (16%)
  - 2017-08-21 23:00 to 2017-10-19 04:30
- Test: 10,484 rows (20%)
  - 2017-10-19 04:40 to 2017-12-30 23:50

## TabM preparation
- Arrays are saved as float32.
- Generated arrays are stored under data/processed/ and are not committed to Git.
- The preparation script recreates the processed dataset from the original UCI source.
- Predictor arrays are saved before model-specific numerical preprocessing so that MLP and TabM can use the same downstream preprocessing pipeline.

## Presentation points
Important changes/decisions to mention:
1. The downloaded dataset contains 52,416 rows rather than the 52,417 stated in the proposal.
2. Zone 1 was selected as the single regression target.
3. The other two zone-consumption variables were excluded as predictors.
4. DateTime was converted into cyclical numerical features.
5. A chronological split was used because observations are time ordered.
6. The final prepared dataset contains 11 numerical predictors.
