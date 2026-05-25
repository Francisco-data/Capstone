"""
Create a preprocessed dataset from the M5 raw data.
Use --all if you to process every row in sales_train_validation.csv (this store a parquet instead of a csv)

Sample Example:
    python preprocess_sample.py --data-dir data/m5/raw --output data/m5/preprocessed_sample.csv --num-items 2

Full dataset example:
    python preprocess_sample.py --data-dir data/m5/raw --output data/m5/preprocessed_all.parquet --all
"""

import argparse
import os

import pandas as pd


def get_output_path(output_path, use_all):
    if not use_all:
        return output_path

    file_name, file_extension = os.path.splitext(output_path)
    if file_extension.lower() == ".parquet":
        return output_path

    parquet_path = f"{file_name}.parquet"
    print(f"--all was used, so the output will be saved as Parquet: {parquet_path}")
    return parquet_path


def preprocess_sample(data_dir, output_path, num_items, use_all) -> None:
    """Load, reshape, merge, clean, and save M5 data."""
    sales_file = os.path.join(data_dir, "sales_train_validation.csv")
    calendar_file = os.path.join(data_dir, "calendar.csv")
    prices_file = os.path.join(data_dir, "sell_prices.csv")

    print("1 -------------- Processing sales file")
    sales = pd.read_csv(sales_file)

    if use_all:
        print("Using all sales rows.")
        print("This take around 5 minutes and creates a Parquet file.")
        sales_to_process = sales
    else:
        # Keep only the first few items sales to keep the output small.
        selected_ids = sales["id"].unique()[:num_items]
        sales_to_process = sales[sales["id"].isin(selected_ids)].copy()
        print(f"Selected {len(selected_ids)} item(s).")

    print("Sales data shape:", sales_to_process.shape)

    # The M5 sales file is wide: d_1, d_2, d_3, etc. are separate columns
    # Melt turn columns into rows, to have a long format
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [
        column for column in sales_to_process.columns if column.startswith("d_")
    ]

    sales_long = sales_to_process.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    print("Long sales shape:", sales_long.shape)

    print("2 -------------- Processing calendar file")
    calendar = pd.read_csv(calendar_file)
    calendar["date"] = pd.to_datetime(calendar["date"])

    # Add calendar columns such as date, weekday, month, and wm_yr_wk.
    merged = sales_long.merge(calendar, on="d", how="left")

    print("3 -------------- Processing price file")
    prices = pd.read_csv(prices_file)

    # Add sell_price using the item, store, and Walmart week.
    merged = merged.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    # Some early days can have missing prices. Fill those prices inside each
    # item/store group using nearby available prices.
    merged["price_missing_flag"] = merged["sell_price"].isna().astype(int)

    merged["sell_price"] = merged.groupby(["store_id", "item_id"])[
        "sell_price"
    ].transform(lambda prices_for_item: prices_for_item.ffill().bfill())

    merged = merged.sort_values(["id", "date"])

    output_path = get_output_path(output_path, use_all)
    output_folder = os.path.dirname(output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    if use_all:
        merged.to_parquet(output_path, index=False)
    else:
        merged.to_csv(output_path, index=False)

    print("Finished.")
    print(f"Saved {len(merged)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a preprocessed sample from the M5 raw data."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing the raw M5 CSV files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to save the preprocessed file. Samples use CSV. --all uses Parquet.",
    )
    item_group = parser.add_mutually_exclusive_group()
    item_group.add_argument(
        "--num-items",
        type=int,
        default=2,
        help="Number of item rows to sample from the sales file. Default: 2.",
    )
    item_group.add_argument(
        "--all",
        action="store_true",
        help="Use every row from sales_train_validation.csv instead of a small sample.",
    )

    args = parser.parse_args()
    if args.num_items is not None and args.num_items < 1:
        parser.error("--num-items must be 1 or greater.")
    return args


if __name__ == "__main__":
    args = parse_args()
    preprocess_sample(args.data_dir, args.output, args.num_items, args.all)
