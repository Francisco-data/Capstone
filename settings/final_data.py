# Data preparation

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

from .final_config import (
    BATCH_SIZE,
    COMBINED_SPLIT_PATH,
    EXPECTED_SERIES_COUNT,
    FEATURE_PATH,
    MAX_EVAL_SERIES_PER_GROUP,
    MAX_TRAIN_SERIES_PER_GROUP,
    SEARCH_SEED,
    STATIC_METADATA_COLUMNS,
    TRAIN_GROUPS,
)

EXCLUSIVE_GROUPS = ["normal", "intermittent", "cold_start_item", "cold_start_store"]


def sample_ids(series_ids, max_count, seed):
    # If max_count is 0, None, or larger than the data, use all series
    series_ids = list(series_ids)

    if max_count is None or max_count <= 0 or max_count >= len(series_ids):
        return series_ids

    return (
        pd
        .Series(series_ids)
        .sample(n=max_count, random_state=seed)
        .astype(int)
        .tolist()
    )


def load_existing_split(split_path=COMBINED_SPLIT_PATH):
    # Read the existing series split.
    series_table = pd.read_csv(split_path)

    # Check that all M5 series are present.
    if len(series_table) != EXPECTED_SERIES_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SERIES_COUNT} series, found {len(series_table)}."
        )

    return series_table


def build_split_ids(
    series_table,
    max_train_per_group=MAX_TRAIN_SERIES_PER_GROUP,
    max_eval_per_group=MAX_EVAL_SERIES_PER_GROUP,
    seed=SEARCH_SEED,
):
    split_ids = {}

    # Prepare IDs for each evaluation group
    for group_name in EXCLUSIVE_GROUPS:
        group_ids = (
            series_table[series_table["exclusive_split"] == group_name]["series_idx"]
            .dropna()
            .astype(int)
            .tolist()
        )

        split_ids[group_name] = sample_ids(
            group_ids,
            max_eval_per_group,
            seed,
        )

    # Combine all exclusive evaluation groups
    split_ids["all_combined"] = []

    for group_name in EXCLUSIVE_GROUPS:
        split_ids["all_combined"].extend(split_ids[group_name])

    # Training uses only normal and intermittent series
    train_ids = []

    for group_name in TRAIN_GROUPS:
        group_ids = (
            series_table[series_table["exclusive_split"] == group_name]["series_idx"]
            .dropna()
            .astype(int)
            .tolist()
        )

        train_ids.extend(
            sample_ids(
                group_ids,
                max_train_per_group,
                seed,
            )
        )

    return train_ids, split_ids


def _unique_columns(columns):
    return list(dict.fromkeys(columns))


def causal_price_values(frame, price_fallback=None):
    # Replace originally missing prices with NaN
    prices = frame["sell_price"].mask(frame["price_missing_flag"] == 1)

    # Use the median training price when no earlier price exists
    if price_fallback is None:
        price_fallback = prices.median()

        if pd.isna(price_fallback):
            price_fallback = 0.0

    # Fill each missing price with the previous price from the same series
    prices = prices.groupby(frame["series_idx"]).ffill()

    # Fill prices at the beginning of a series with the fallback
    prices = prices.fillna(price_fallback)

    return prices, float(price_fallback)


def fit_covariate_scaler(
    series_ids,
    covariate_columns,
    fit_end_idx,
    feature_path=FEATURE_PATH,
):
    # Return no scaler when the model does not use covariates
    if not covariate_columns:
        return None, None

    read_columns = [
        "series_idx",
        "time_idx",
        *covariate_columns,
    ]

    if "sell_price" in covariate_columns:
        read_columns.append("price_missing_flag")

    # Remove duplicated column names
    read_columns = list(dict.fromkeys(read_columns))

    # Filter series
    scaler_df = pd.read_parquet(
        feature_path,
        columns=read_columns,
        filters=[
            ("series_idx", "in", list(series_ids)),
            ("time_idx", "<=", fit_end_idx),
        ],
    )

    scaler_df = scaler_df.sort_values(["series_idx", "time_idx"])

    price_fallback = None

    if "sell_price" in covariate_columns:
        scaler_df["sell_price"], price_fallback = causal_price_values(scaler_df)

    # Replace invalid values before fitting the scaler
    covariate_values = (
        scaler_df[covariate_columns]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    scaler = RobustScaler()
    scaler.fit(covariate_values)

    return scaler, price_fallback


def fit_metadata_encoders(
    series_table,
    series_ids,
    metadata_columns=STATIC_METADATA_COLUMNS,
):
    # Use metadata from training series
    training_metadata = series_table[series_table["series_idx"].isin(series_ids)]

    encoders = {}
    cardinalities = []

    for column in metadata_columns:
        known_values = sorted(training_metadata[column].dropna().astype(str).unique())

        # Index 0 is for unknown categories
        encoders[column] = {
            value: index + 1 for index, value in enumerate(known_values)
        }

        cardinalities.append(len(known_values) + 1)

    return encoders, cardinalities


def load_series_arrays(
    series_ids,
    series_table,
    required_end_idx,
    past_covariate_columns=None,
    future_covariate_columns=None,
    metadata_columns=None,
    covariate_scaler=None,
    price_fallback=None,
    metadata_encoders=None,
    target_column="log_sales",
    feature_path=FEATURE_PATH,
):
    past_covariate_columns = past_covariate_columns or []
    future_covariate_columns = future_covariate_columns or []
    metadata_columns = metadata_columns or []
    metadata_encoders = metadata_encoders or {}

    # Combine covariate columns without duplicates
    all_covariate_columns = list(
        dict.fromkeys(past_covariate_columns + future_covariate_columns)
    )

    read_columns = [
        "series_idx",
        "time_idx",
        "sales",
        target_column,
        *all_covariate_columns,
    ]

    if "sell_price" in all_covariate_columns:
        read_columns.append("price_missing_flag")

    read_columns = list(dict.fromkeys(read_columns))

    frame = pd.read_parquet(
        feature_path,
        columns=read_columns,
        filters=[("series_idx", "in", list(series_ids))],
    )

    frame = frame.sort_values(["series_idx", "time_idx"])
    metadata_lookup = series_table.set_index("series_idx")

    past_indices = [
        all_covariate_columns.index(column) for column in past_covariate_columns
    ]
    future_indices = [
        all_covariate_columns.index(column) for column in future_covariate_columns
    ]

    loaded_data = {
        "past_arrays": [],
        "future_arrays": [],
        "sales_arrays": [],
        "metadata_arrays": [],
        "series_ids": [],
    }

    for series_idx, group in frame.groupby("series_idx", sort=True):
        group = group.sort_values("time_idx").copy()

        # Skip series without enough observations
        if len(group) <= required_end_idx:
            continue

        if "sell_price" in all_covariate_columns:
            group["sell_price"], _ = causal_price_values(
                group,
                price_fallback,
            )

        # Scale numerical covariates
        if all_covariate_columns:
            covariate_values = (
                group[all_covariate_columns]
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )

            covariate_values = covariate_scaler.transform(covariate_values).astype(
                np.float32
            )
        else:
            covariate_values = np.empty(
                (len(group), 0),
                dtype=np.float32,
            )

        log_sales = group[[target_column]].to_numpy(dtype=np.float32)
        sales = group["sales"].to_numpy(dtype=np.float32)

        past_covariates = covariate_values[:, past_indices]
        future_covariates = covariate_values[:, future_indices]

        # Channel 0 is log_sales.
        past_values = np.concatenate(
            [log_sales, past_covariates],
            axis=1,
        )

        metadata = np.array(
            [
                metadata_encoders[column].get(
                    str(metadata_lookup.loc[series_idx, column]),
                    0,
                )
                for column in metadata_columns
            ],
            dtype=np.int64,
        )

        loaded_data["past_arrays"].append(past_values)
        loaded_data["future_arrays"].append(future_covariates)
        loaded_data["sales_arrays"].append(sales)
        loaded_data["metadata_arrays"].append(metadata)
        loaded_data["series_ids"].append(int(series_idx))

    return loaded_data


def select_loaded_series(loaded_data, selected_ids):
    # Find the position of every loaded series
    position_by_id = {
        series_idx: position
        for position, series_idx in enumerate(loaded_data["series_ids"])
    }

    selected_positions = [
        position_by_id[series_idx]
        for series_idx in selected_ids
        if series_idx in position_by_id
    ]

    return {
        "past_arrays": [
            loaded_data["past_arrays"][position] for position in selected_positions
        ],
        "future_arrays": [
            loaded_data["future_arrays"][position] for position in selected_positions
        ],
        "sales_arrays": [
            loaded_data["sales_arrays"][position] for position in selected_positions
        ],
        "metadata_arrays": [
            loaded_data["metadata_arrays"][position] for position in selected_positions
        ],
        "series_ids": [
            loaded_data["series_ids"][position] for position in selected_positions
        ],
    }


class RandomWindowDataset(Dataset):
    def __init__(
        self,
        loaded_data,
        seq_len,
        pred_len,
        training_end_exclusive,
        samples_per_epoch,
        seed=SEARCH_SEED,
    ):
        self.data = loaded_data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.training_end = training_end_exclusive
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_epoch(self, epoch):
        # reproducible windows each epoch
        self.rng = np.random.default_rng(self.seed + epoch)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        series_position = self.rng.integers(len(self.data["past_arrays"]))

        past = self.data["past_arrays"][series_position]
        future = self.data["future_arrays"][series_position]
        metadata = self.data["metadata_arrays"][series_position]

        max_start = self.training_end - self.seq_len - self.pred_len
        start = self.rng.integers(max_start + 1)

        target_start = start + self.seq_len
        target_end = target_start + self.pred_len

        return {
            "x_past": torch.from_numpy(past[start:target_start].copy()),
            "x_future": torch.from_numpy(future[target_start:target_end].copy()),
            "y_log": torch.from_numpy(past[target_start:target_end, 0:1].copy()),
            "metadata": torch.from_numpy(metadata.copy()),
        }


def build_train_loader(
    loaded_data,
    seq_len,
    pred_len,
    training_end_exclusive,
    samples_per_epoch,
    seed=SEARCH_SEED,
    batch_size=BATCH_SIZE,
):
    dataset = RandomWindowDataset(
        loaded_data,
        seq_len,
        pred_len,
        training_end_exclusive,
        samples_per_epoch,
        seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def build_forecast_data(
    loaded_data,
    seq_len,
    forecast_start_idx,
    pred_len,
):
    start = forecast_start_idx
    end = start + pred_len

    return {
        "x_past": np.stack([
            values[start - seq_len : start] for values in loaded_data["past_arrays"]
        ]).astype(np.float32),
        "x_future": np.stack([
            values[start:end] for values in loaded_data["future_arrays"]
        ]).astype(np.float32),
        "metadata": np.stack(loaded_data["metadata_arrays"]).astype(np.int64),
        "actual_sales": np.stack([
            values[start:end] for values in loaded_data["sales_arrays"]
        ]).astype(np.float32),
        "seasonal_naive": np.stack([
            values[start - pred_len : start] for values in loaded_data["sales_arrays"]
        ]).astype(np.float32),
        "series_ids": np.array(
            loaded_data["series_ids"],
            dtype=np.int64,
        ),
        "forecast_start_idx": start,
    }


def save_preprocessors(
    output_dir,
    scaler=None,
    price_fallback=None,
    metadata_encoders=None,
    prefix="search",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if scaler is not None:
        with open(
            output_dir / f"{prefix}_covariate_scaler.pkl",
            "wb",
        ) as file:
            pickle.dump(scaler, file)

    preprocessing_data = {
        "price_fallback": price_fallback,
        "metadata_encoders": metadata_encoders or {},
    }

    with open(
        output_dir / f"{prefix}_preprocessing_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(preprocessing_data, file, indent=2)


def load_preprocessors(output_dir, prefix="search"):
    # Load the scaler when the experiment uses numerical covariates.
    output_dir = Path(output_dir)
    scaler_path = output_dir / f"{prefix}_covariate_scaler.pkl"
    scaler = None

    if scaler_path.exists():
        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)

    # Load the price fallback and metadata encoders.
    with open(
        output_dir / f"{prefix}_preprocessing_metadata.json",
        encoding="utf-8",
    ) as file:
        preprocessing_data = json.load(file)

    return (
        scaler,
        preprocessing_data["price_fallback"],
        preprocessing_data["metadata_encoders"],
    )
