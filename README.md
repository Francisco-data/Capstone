# Unified Forecasting Project

This project repository is structured to facilitate a week 0 setup for your unified forecasting project. It includes a clear folder hierarchy, example code to preprocess a small sample of the M5 dataset, and basic environment configuration files. You should complete this setup before starting the main research work outlined in your project planning document.

## Directory structure

```text
unified_forecasting/
├── README.md
├── pyproject.toml
├── preprocess_sample.py
└── data/
    └── m5/
        ├── raw/
        │   └── (place the original M5 CSV files here)
        └── preprocessed_sample.csv
```

* `pyproject.toml` lists the core Python dependencies used for the preprocessing script and baseline experiments.
* `preprocess_sample.py` is a Python script that reads a few rows from the M5 dataset, merges sales, calendar, and price tables, performs basic cleaning, and writes a small sample to `data/m5/preprocessed_sample.csv`. You can run this script once the M5 dataset is available in the `raw` directory.
* `data/m5/raw/` should contain the original M5 files downloaded from Kaggle (`calendar.csv`, `sell_prices.csv`, `sales_train_validation.csv`, etc.). Do not commit large data files to version control.
* `data/m5/preprocessed_sample.csv` is a placeholder file that you can overwrite with the output of `preprocess_sample.py`.

## Setup instructions

1. Create a Python virtual environment with uv:

   ```bash
   uv venv
   ```

2. Sync the project dependencies:

   ```bash
   uv sync
   ```

3. Download the M5 dataset from Kaggle and place the CSV files into `data/m5/raw/`.

4. Run the preprocessing script:

   ```bash
   uv run python preprocess_sample.py --data-dir data/m5/raw --output data/m5/preprocessed_sample.csv --num-items 2
   ```

The `--num-items` parameter controls how many items are sampled from the dataset for quick experiments. Once the script completes, you should have a `preprocessed_sample.csv` file ready for initial exploratory analysis and baseline experiments.

To preprocess every sales row from `sales_train_validation.csv`, use `--all`:

```bash
uv run python preprocess_sample.py --data-dir data/m5/raw --output data/m5/preprocessed_all.parquet --all
```

This full output can be very large because the sales file has many item/store rows and many daily sales columns. Parquet is used for the full output because it is usually smaller and faster to reload than CSV.
