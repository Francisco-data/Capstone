"""
Create split metadata for M5 evaluation groups.

This script reads the feature-engineered M5 long table and saves small CSV files
that identify intermittent series, cold-start item series, cold-start store
series, and normal series.
It also saves one combined exclusive split where each series belongs to exactly
one group.

Example:
    python data/splits.py --input data/m5/processed/m5_long_features.parquet --output-dir data/m5/processed/splits
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REQUIRED_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "series_idx",
    "item_idx",
    "store_idx",
    "sales",
    "zero_demand_flag",
    "date",
]

SERIES_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "series_idx",
    "item_idx",
    "store_idx",
    "start_date",
    "end_date",
    "total_days",
    "zero_days",
    "zero_ratio",
    "nonzero_days",
    "total_sales",
    "mean_sales",
    "is_intermittent",
    "is_cold_start_item",
    "is_cold_start_store",
    "is_normal",
]

EXCLUSIVE_SERIES_COLUMNS = SERIES_COLUMNS + [
    "exclusive_split",
    "is_exclusive_cold_start_item",
    "is_exclusive_cold_start_store",
    "is_exclusive_intermittent",
    "is_exclusive_normal",
]


def load_feature_columns(input_path: Path) -> pd.DataFrame:
    """Load only the columns needed to generate split metadata."""
    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(input_path, columns=REQUIRED_COLUMNS)

    if suffix == ".csv":
        return pd.read_csv(input_path, usecols=REQUIRED_COLUMNS, parse_dates=["date"])

    raise ValueError("Input file must be .csv or .parquet")


def check_required_columns(input_path: Path) -> None:
    """Check that the input file has all columns required for split metadata."""
    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        available_columns = pq.ParquetFile(input_path).schema_arrow.names
    elif suffix == ".csv":
        available_columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    else:
        raise ValueError("Input file must be .csv or .parquet")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in available_columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Input data is missing required columns: {missing_text}")


def create_series_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the long feature table into one row per series."""
    series_metadata = (
        df
        .groupby(
            [
                "id",
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
                "series_idx",
                "item_idx",
                "store_idx",
            ],
            observed=True,
        )
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            total_days=("sales", "size"),
            zero_days=("zero_demand_flag", "sum"),
            total_sales=("sales", "sum"),
            mean_sales=("sales", "mean"),
        )
        .reset_index()
    )

    series_metadata["zero_ratio"] = (
        series_metadata["zero_days"] / series_metadata["total_days"]
    )
    series_metadata["nonzero_days"] = (
        series_metadata["total_days"] - series_metadata["zero_days"]
    )

    return series_metadata


def choose_holdouts(values: pd.Series, count: int, seed: int) -> list[str]:
    """Choose reproducible holdout values from sorted unique IDs."""
    unique_values = sorted(values.dropna().unique())
    rng = np.random.default_rng(seed)
    selected_values = rng.choice(unique_values, size=count, replace=False)
    return sorted(selected_values.tolist())


def add_split_flags(
    series_metadata: pd.DataFrame,
    intermittent_threshold: float,
    heldout_items: list[str],
    heldout_stores: list[str],
) -> pd.DataFrame:
    """Add split flags to the series metadata."""
    # Intermittent demand means the series has many zero-sales days
    series_metadata["is_intermittent"] = (
        series_metadata["zero_ratio"] >= intermittent_threshold
    ).astype("int8")

    # Cold-start item/store flags mark series that should be hidden for those
    # evaluation cases
    series_metadata["is_cold_start_item"] = (
        series_metadata["item_id"].isin(heldout_items).astype("int8")
    )
    series_metadata["is_cold_start_store"] = (
        series_metadata["store_id"].isin(heldout_stores).astype("int8")
    )
    series_metadata["is_normal"] = (
        (series_metadata["zero_ratio"] < intermittent_threshold)
        & (series_metadata["is_cold_start_item"] == 0)
        & (series_metadata["is_cold_start_store"] == 0)
    ).astype("int8")

    return series_metadata


def add_exclusive_split(series_metadata: pd.DataFrame) -> pd.DataFrame:
    """Assign each series to one split using a fixed priority order."""
    # Start with normal as the default group.
    series_metadata["exclusive_split"] = "normal"

    # Apply the groups in reverse priority order. Later assignments overwrite
    # earlier ones, so cold-start item has the highest priority.
    series_metadata.loc[
        series_metadata["is_intermittent"] == 1,
        "exclusive_split",
    ] = "intermittent"
    series_metadata.loc[
        series_metadata["is_cold_start_store"] == 1,
        "exclusive_split",
    ] = "cold_start_store"
    series_metadata.loc[
        series_metadata["is_cold_start_item"] == 1,
        "exclusive_split",
    ] = "cold_start_item"

    # Add one-hot style flags for quick filtering
    series_metadata["is_exclusive_cold_start_item"] = (
        series_metadata["exclusive_split"] == "cold_start_item"
    ).astype("int8")
    series_metadata["is_exclusive_cold_start_store"] = (
        series_metadata["exclusive_split"] == "cold_start_store"
    ).astype("int8")
    series_metadata["is_exclusive_intermittent"] = (
        series_metadata["exclusive_split"] == "intermittent"
    ).astype("int8")
    series_metadata["is_exclusive_normal"] = (
        series_metadata["exclusive_split"] == "normal"
    ).astype("int8")

    return series_metadata


def save_split_files(
    series_metadata: pd.DataFrame,
    heldout_items: list[str],
    heldout_stores: list[str],
    output_dir: Path,
) -> None:
    """Save separate CSV files for each evaluation group."""
    output_dir.mkdir(parents=True, exist_ok=True)

    series_metadata.loc[series_metadata["is_intermittent"] == 1, SERIES_COLUMNS].to_csv(
        output_dir / "intermittent_series.csv", index=False
    )

    series_metadata.loc[
        series_metadata["is_cold_start_item"] == 1, SERIES_COLUMNS
    ].to_csv(output_dir / "cold_start_item_series.csv", index=False)

    series_metadata.loc[
        series_metadata["is_cold_start_store"] == 1, SERIES_COLUMNS
    ].to_csv(output_dir / "cold_start_store_series.csv", index=False)

    series_metadata.loc[series_metadata["is_normal"] == 1, SERIES_COLUMNS].to_csv(
        output_dir / "normal_series.csv", index=False
    )

    heldout_item_rows = (
        series_metadata
        .loc[series_metadata["item_id"].isin(heldout_items), ["item_id", "item_idx"]]
        .drop_duplicates()
        .sort_values("item_id")
    )
    heldout_item_rows.to_csv(output_dir / "heldout_items.csv", index=False)

    heldout_store_rows = (
        series_metadata
        .loc[
            series_metadata["store_id"].isin(heldout_stores), ["store_id", "store_idx"]
        ]
        .drop_duplicates()
        .sort_values("store_id")
    )
    heldout_store_rows.to_csv(output_dir / "heldout_stores.csv", index=False)

    series_metadata[EXCLUSIVE_SERIES_COLUMNS].to_csv(
        output_dir / "combined_exclusive_series.csv",
        index=False,
    )


def run_split_pipeline(
    input_path: Path,
    output_dir: Path,
    intermittent_threshold: float,
    item_holdout_fraction: float,
    store_holdout_count: int,
    seed: int,
) -> None:
    """Create all split metadata files."""
    print(f"Checking input columns: {input_path}")
    check_required_columns(input_path)

    print("Loading required feature columns...")
    df = load_feature_columns(input_path)
    print("Input shape:", df.shape)

    print("Creating series-level metadata...")
    series_metadata = create_series_metadata(df)
    print("Series count:", len(series_metadata))

    item_holdout_count = math.ceil(
        series_metadata["item_id"].nunique() * item_holdout_fraction
    )
    heldout_items = choose_holdouts(
        series_metadata["item_id"], item_holdout_count, seed
    )
    heldout_stores = choose_holdouts(
        series_metadata["store_id"], store_holdout_count, seed
    )

    print("Held-out item count:", len(heldout_items))
    print("Held-out store count:", len(heldout_stores))

    series_metadata = add_split_flags(
        series_metadata,
        intermittent_threshold,
        heldout_items,
        heldout_stores,
    )
    series_metadata = add_exclusive_split(series_metadata)

    save_split_files(series_metadata, heldout_items, heldout_stores, output_dir)

    print("Saved split metadata to:", output_dir)
    print("Intermittent series:", int(series_metadata["is_intermittent"].sum()))
    print("Cold-start item series:", int(series_metadata["is_cold_start_item"].sum()))
    print("Cold-start store series:", int(series_metadata["is_cold_start_store"].sum()))
    print("Normal series:", int(series_metadata["is_normal"].sum()))
    print("Exclusive split counts:")
    print(series_metadata["exclusive_split"].value_counts().sort_index().to_string())
    print("Finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create M5 split metadata for intermittent and cold-start evaluation."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input feature-engineered M5 file (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where split metadata CSV files will be saved.",
    )
    parser.add_argument(
        "--intermittent-threshold",
        type=float,
        default=0.60,
        help="Zero-ratio threshold for intermittent series. Default: 0.60.",
    )
    parser.add_argument(
        "--item-holdout-fraction",
        type=float,
        default=0.10,
        help="Fraction of item_id values to hold out. Default: 0.10.",
    )
    parser.add_argument(
        "--store-holdout-count",
        type=int,
        default=1,
        help="Number of store_id values to hold out. Default: 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible holdouts. Default: 42.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_split_pipeline(
        args.input,
        args.output_dir,
        args.intermittent_threshold,
        args.item_holdout_fraction,
        args.store_holdout_count,
        args.seed,
    )
