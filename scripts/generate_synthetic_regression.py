
from pathlib import Path
import numpy as np
import json

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

train_sizes = [500, 2000, 8000]
noise_levels = [0.0, 0.10, 0.30]

n_val = 2000
n_test = 4000
n_features = 10
max_train = max(train_sizes)

repo_root = Path(__file__).resolve().parents[1]
output_root = repo_root / "data" / "processed" / "synthetic-regression"
output_root.mkdir(parents=True, exist_ok=True)

total_n = max_train + n_val + n_test

X_all = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(total_n, n_features)
)

signal = (
    3.0 * X_all[:, 0]
    - 2.0 * X_all[:, 1]
    + 1.5 * X_all[:, 2] * X_all[:, 3]
    + 2.0 * np.sin(X_all[:, 4])
    + 0.8 * (X_all[:, 5] ** 2)
    - 1.2 * X_all[:, 6]
)

signal_std = signal.std()

train_pool_X = X_all[:max_train]
train_pool_signal = signal[:max_train]

X_val = X_all[max_train:max_train + n_val]
signal_val = signal[max_train:max_train + n_val]

X_test = X_all[max_train + n_val:]
signal_test = signal[max_train + n_val:]

for train_size in train_sizes:
    for noise_level in noise_levels:

        condition_seed = (
            RANDOM_SEED
            + train_size
            + int(noise_level * 1000)
        )

        condition_rng = np.random.default_rng(condition_seed)
        noise_sd = noise_level * signal_std

        X_train = train_pool_X[:train_size]

        y_train = (
            train_pool_signal[:train_size]
            + condition_rng.normal(0, noise_sd, train_size)
        )

        y_val = (
            signal_val
            + condition_rng.normal(0, noise_sd, n_val)
        )

        y_test = (
            signal_test
            + condition_rng.normal(0, noise_sd, n_test)
        )

        dataset_id = (
            f"synthetic-n{train_size}-noise{int(noise_level * 100)}"
        )

        dataset_dir = output_root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        np.save(dataset_dir / "X_num_train.npy", X_train.astype(np.float32))
        np.save(dataset_dir / "X_num_val.npy", X_val.astype(np.float32))
        np.save(dataset_dir / "X_num_test.npy", X_test.astype(np.float32))

        np.save(dataset_dir / "Y_train.npy", y_train.astype(np.float32))
        np.save(dataset_dir / "Y_val.npy", y_val.astype(np.float32))
        np.save(dataset_dir / "Y_test.npy", y_test.astype(np.float32))

        info = {
            "task_type": "regression",
            "name": dataset_id,
            "id": dataset_id
        }

        with open(dataset_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        (dataset_dir / "READY").touch()

print("Synthetic datasets recreated: 9")
print("Signal standard deviation:", round(signal_std, 4))
