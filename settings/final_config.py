from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data/m5/processed/m5_long_features.parquet"
COMBINED_SPLIT_PATH = (
    PROJECT_ROOT / "data/m5/processed/splits/combined_exclusive_series.csv"
)

RESULT_ROOT = PROJECT_ROOT / "notebooks/experiments"

# Intermittent labels use all 1,913 days
INTERMITTENT_LABEL_PERIOD = "d_1 to d_1913"
EXPECTED_SERIES_COUNT = 30_490

# train, validation, and final evaluation indexes
SEARCH_TRAIN_END_IDX = 1856
VALIDATION_START_IDX = 1857
VALIDATION_END_IDX = 1884
TEST_START_IDX = 1885
TEST_END_IDX = 1912

PRED_LEN = 28
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 1024
SEARCH_SEED = 42
FINAL_SEEDS = [42, 300, 2026, 7, 1234]

TRAIN_GROUPS = ["normal", "intermittent"]
EVAL_GROUPS = [
    "normal",
    "intermittent",
    "cold_start_item",
    "cold_start_store",
    "all_combined",
]

# 0 means use all available series
MAX_TRAIN_SERIES_PER_GROUP = 0
MAX_EVAL_SERIES_PER_GROUP = 0

CONVERGENCE_MIN_DELTA = 1e-5
ZERO_SALES_THRESHOLD = 0.5
NONZERO_PROBABILITY_THRESHOLD = 0.5
OCCURRENCE_LOSS_WEIGHT = 1.0
MAGNITUDE_LOSS_WEIGHT = 1.0
EMBEDDING_DIM = 4

PAST_COVARIATE_COLUMNS = [
    "day_of_week",
    "is_weekend",
    "event_flag",
    "snap_flag",
    "sell_price",
]

# Future-covariate model only
FUTURE_KNOWN_COVARIATE_COLUMNS = [
    "day_of_week",
    "is_weekend",
    "event_flag",
    "snap_flag",
]

STATIC_METADATA_COLUMNS = [
    "cat_id",
    "dept_id",
    "state_id",
]

# Full model exclude future covariates component
FULL_MODEL_FUTURE_COVARIATE_COLUMNS = []

GRID = {
    "seq_len": [28, 56, 112],
    "learning_rate": [1e-3, 3e-4],
    "epochs": [50],
    "samples_per_epoch": [10_000, 20_000, 40_000],
}
