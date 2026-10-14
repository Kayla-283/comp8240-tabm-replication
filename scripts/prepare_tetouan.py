
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


# Repository paths
repo_root = Path(__file__).resolve().parents[1]
output_dir = repo_root / "data" / "processed" / "tetouan-zone1"
output_dir.mkdir(parents=True, exist_ok=True)


# Download official UCI dataset
tetouan = fetch_ucirepo(id=849)

X = tetouan.data.features.copy()
y = tetouan.data.targets.copy()

df = pd.concat([X, y], axis=1)


# Convert DateTime to a proper datetime variable
df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    format="%m/%d/%Y %H:%M"
)


# Create cyclical time features
minutes_in_day = (
    df["DateTime"].dt.hour * 60
    + df["DateTime"].dt.minute
)

df["time_sin"] = np.sin(2 * np.pi * minutes_in_day / 1440)
df["time_cos"] = np.cos(2 * np.pi * minutes_in_day / 1440)

day_of_week = df["DateTime"].dt.dayofweek
df["weekday_sin"] = np.sin(2 * np.pi * day_of_week / 7)
df["weekday_cos"] = np.cos(2 * np.pi * day_of_week / 7)

day_of_year = df["DateTime"].dt.dayofyear
df["year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
df["year_cos"] = np.cos(2 * np.pi * day_of_year / 365)


# Define predictors and Zone 1 regression target
feature_columns = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
    "time_sin",
    "time_cos",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
]

target_column = "Zone 1 Power Consumption"

X_model = df[feature_columns]
y_model = df[target_column]


# Chronological 64% / 16% / 20% split
n = len(df)

train_end = int(n * 0.64)
val_end = int(n * 0.80)

X_train = X_model.iloc[:train_end]
X_val = X_model.iloc[train_end:val_end]
X_test = X_model.iloc[val_end:]

y_train = y_model.iloc[:train_end]
y_val = y_model.iloc[train_end:val_end]
y_test = y_model.iloc[val_end:]


# Save arrays in the format expected by the TabM repository
np.save(output_dir / "X_num_train.npy", X_train.to_numpy(dtype=np.float32))
np.save(output_dir / "X_num_val.npy", X_val.to_numpy(dtype=np.float32))
np.save(output_dir / "X_num_test.npy", X_test.to_numpy(dtype=np.float32))

np.save(output_dir / "Y_train.npy", y_train.to_numpy(dtype=np.float32))
np.save(output_dir / "Y_val.npy", y_val.to_numpy(dtype=np.float32))
np.save(output_dir / "Y_test.npy", y_test.to_numpy(dtype=np.float32))


# TabM dataset metadata
info = {
    "task_type": "regression",
    "name": "Power Consumption of Tetouan City - Zone 1",
    "id": "tetouan-zone1",
}

with open(output_dir / "info.json", "w") as f:
    json.dump(info, f, indent=2)

(output_dir / "READY").touch()


# Reproducibility summary
print("Tetouan preparation complete.")
print("Total rows:", n)
print("Predictors:", len(feature_columns))
print("Train rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
print("Output:", output_dir)
