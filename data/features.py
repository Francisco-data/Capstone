"""
Creates extra fields for the M5 data, including
    categorical mappings,
            "item_id": "item_idx",
            "dept_id": "dept_idx",
            "cat_id": "cat_idx",
            "store_id": "store_idx",
            "state_id": "state_idx",
            "id": "series_idx"
    calendar features,
            "day_of_week",
            "is_weekend",
            "time_idx"
    event flags,
            "event_flag",
            "snap_flag"
    price change features,
            "price_change_pct"
    demand features.
            "zero_demand_flag",
            "log_sales"

Example:
    python data/features.py --input data/m5/preprocessed_all.parquet --output data/m5/processed/m5_long_features.parquet
    python data/features.py --input data/m5/preprocessed_all.parquet --output data/m5/processed/m5_long_features.csv

**Mappings are saved in a mappings/ subdirectory
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORY_COLUMNS = {
    "item_id": "item_idx",
    "dept_id": "dept_idx",
    "cat_id": "cat_idx",
    "store_id": "store_idx",
    "state_id": "state_idx",
    "id": "series_idx",
}

MAPPING_FILE_NAMES = {
    "item_id": "item_mapping.csv",
    "dept_id": "dept_mapping.csv",
    "cat_id": "cat_mapping.csv",
    "store_id": "store_mapping.csv",
    "state_id": "state_mapping.csv",
    "id": "series_mapping.csv",
}

REQUIRED_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "sales",
    "date",
    "event_name_1",
    "event_name_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sell_price",
    "price_missing_flag",
]


def load_long_data(input_path: Path) -> pd.DataFrame:
    """Load a preprocessed M5 file. csv or parquet."""
    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path, parse_dates=["date"])
    else:
        raise ValueError("Input file must be .csv or .parquet")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    return df


def check_required_columns(df: pd.DataFrame) -> None:
    """Validation of required columns before creating features."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Input data is missing required columns: {missing_text}")


def save_mapping(
    df: pd.DataFrame,
    category_column: str,
    index_column: str,
    mappings_dir: Path,
) -> pd.DataFrame:
    """Create one mapping and save it as a CSV file."""
    unique_values = sorted(df[category_column].dropna().unique())
    mapping = pd.DataFrame(
        {
            category_column: unique_values,
            index_column: range(len(unique_values)),
        }
    )

    mapping_path = mappings_dir / MAPPING_FILE_NAMES[category_column]
    mapping.to_csv(mapping_path, index=False)
    return mapping


def add_mapping_columns(df: pd.DataFrame, mappings_dir: Path) -> pd.DataFrame:
    """Add integer index columns for each categorical identifier."""
    for category_column, index_column in CATEGORY_COLUMNS.items():
        mapping = save_mapping(df, category_column, index_column, mappings_dir)
        mapping_dict = dict(zip(mapping[category_column], mapping[index_column]))
        df[index_column] = df[category_column].map(mapping_dict)
        df[index_column] = df[index_column].astype("int32")

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple time-based features."""
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["time_idx"] = df.groupby("id").cumcount().astype("int16")
    return df


def add_event_and_snap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add event and state-specific SNAP flags."""
    has_event_1 = df["event_name_1"].notna()
    has_event_2 = df["event_name_2"].notna()
    df["event_flag"] = (has_event_1 | has_event_2).astype("int8")

    state_snap_columns = {
        "CA": "snap_CA",
        "TX": "snap_TX",
        "WI": "snap_WI",
    }

    df["snap_flag"] = 0
    for state, snap_column in state_snap_columns.items():
        state_rows = df["state_id"] == state
        df.loc[state_rows, "snap_flag"] = df.loc[state_rows, snap_column]

    df["snap_flag"] = df["snap_flag"].astype("int8")
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price change features within each time series."""
    df["price_change_pct"] = (
        df.groupby("id")["sell_price"]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .astype("float32")
    )
    return df


def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand flags, log demand, and simple lag features."""
    df["zero_demand_flag"] = (df["sales"] == 0).astype("int8")
    df["log_sales"] = np.log1p(df["sales"]).astype("float32")

    for lag in [1, 7, 28]:
        df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)

    return df


def create_features(df: pd.DataFrame, mappings_dir: Path) -> pd.DataFrame:
    """Create all features and save mapping CSV files."""
    check_required_columns(df)

    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_event_and_snap_features(df)
    df = add_price_features(df)
    df = add_demand_features(df)
    df = add_mapping_columns(df, mappings_dir)

    return df


def run_feature_pipeline(input_path: Path, output_path: Path) -> None:
    """Load long data, add features, save mappings, and write CSV or Parquet output."""
    output_format = output_path.suffix.lower()
    if output_format not in [".csv", ".parquet"]:
        raise ValueError("Feature output must be a .csv or .parquet file.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mappings_dir = output_path.parent / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input data: {input_path}")
    df = load_long_data(input_path)
    print("Input shape:", df.shape)

    print("Creating features...")
    features = create_features(df, mappings_dir)
    print("Feature shape:", features.shape)

    print(f"Saving feature data: {output_path}")
    if output_format == ".parquet":
        features.to_parquet(output_path, index=False)
    else:
        features.to_csv(output_path, index=False)

    print(f"Saved mappings to: {mappings_dir}")
    print("Finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create feature-engineered M5 data.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input preprocessed M5 file (csv or parquet)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output feature file (csv or parquet)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_feature_pipeline(args.input, args.output)
