"""Evaluation functions shared by the final model notebooks."""

import numpy as np
import pandas as pd
import torch

from .final_config import (
    EVAL_BATCH_SIZE,
    NONZERO_PROBABILITY_THRESHOLD,
    PRED_LEN,
    ZERO_SALES_THRESHOLD,
)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else np.nan


def calculate_metrics(predicted_sales, actual_sales, predicted_zero=None):
    predicted_sales = np.asarray(predicted_sales)
    actual_sales = np.asarray(actual_sales)

    if predicted_zero is None:
        predicted_zero = predicted_sales < ZERO_SALES_THRESHOLD

    actual_zero = actual_sales == 0
    predicted_zero = np.asarray(predicted_zero, dtype=bool)
    absolute_error = np.abs(predicted_sales - actual_sales)

    true_zero_predicted_zero = int(np.sum(actual_zero & predicted_zero))
    true_zero_predicted_nonzero = int(np.sum(actual_zero & ~predicted_zero))
    true_nonzero_predicted_zero = int(np.sum(~actual_zero & predicted_zero))
    true_nonzero_predicted_nonzero = int(np.sum(~actual_zero & ~predicted_zero))

    zero_precision = safe_divide(
        true_zero_predicted_zero,
        true_zero_predicted_zero + true_nonzero_predicted_zero,
    )
    zero_recall = safe_divide(
        true_zero_predicted_zero,
        true_zero_predicted_zero + true_zero_predicted_nonzero,
    )
    zero_f1 = safe_divide(
        2 * zero_precision * zero_recall,
        zero_precision + zero_recall,
    )

    nonzero_precision = safe_divide(
        true_nonzero_predicted_nonzero,
        true_nonzero_predicted_nonzero + true_zero_predicted_nonzero,
    )
    nonzero_recall = safe_divide(
        true_nonzero_predicted_nonzero,
        true_nonzero_predicted_nonzero + true_nonzero_predicted_zero,
    )
    nonzero_f1 = safe_divide(
        2 * nonzero_precision * nonzero_recall,
        nonzero_precision + nonzero_recall,
    )

    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean((predicted_sales - actual_sales) ** 2))),
        "wape": float(
            safe_divide(absolute_error.sum(), np.abs(actual_sales).sum())
        ),
        "zero_precision": float(zero_precision),
        "zero_recall": float(zero_recall),
        "zero_f1": float(zero_f1),
        "nonzero_precision": float(nonzero_precision),
        "nonzero_recall": float(nonzero_recall),
        "nonzero_f1": float(nonzero_f1),
        "balanced_accuracy": float(np.nanmean([zero_recall, nonzero_recall])),
        "true_zero_predicted_zero": true_zero_predicted_zero,
        "true_zero_predicted_nonzero": true_zero_predicted_nonzero,
        "true_nonzero_predicted_zero": true_nonzero_predicted_zero,
        "true_nonzero_predicted_nonzero": true_nonzero_predicted_nonzero,
        "number_of_series": int(actual_sales.shape[0]),
        "number_of_forecast_observations": int(actual_sales.size),
        "zero_ratio": float(np.mean(actual_zero)),
        "total_actual_demand": float(actual_sales.sum()),
        "total_predicted_demand": float(predicted_sales.sum()),
    }


def build_baseline_predictions(forecast_data, pred_len=PRED_LEN):
    last_value = np.expm1(forecast_data["x_past"][:, -1, 0:1])

    return {
        "zero": np.zeros_like(forecast_data["actual_sales"]),
        "last_value": np.repeat(last_value, pred_len, axis=1),
        "seasonal_naive_28": forecast_data["seasonal_naive"],
    }


def build_tensor_batch(forecast_data, start, end, device):
    return {
        "x_past": torch.from_numpy(forecast_data["x_past"][start:end]).to(device),
        "x_future": torch.from_numpy(
            forecast_data["x_future"][start:end]
        ).to(device),
        "metadata": torch.from_numpy(
            forecast_data["metadata"][start:end]
        ).long().to(device),
    }


def predict_regression_model(
    model,
    forecast_data,
    model_forward,
    device,
    eval_batch_size=EVAL_BATCH_SIZE,
):
    model.eval()
    predicted_log_values = []
    number_of_series = len(forecast_data["x_past"])

    with torch.no_grad():
        for start in range(0, number_of_series, eval_batch_size):
            end = min(start + eval_batch_size, number_of_series)
            batch = build_tensor_batch(forecast_data, start, end, device)
            predicted_log = model_forward(model, batch)
            predicted_log_values.append(
                predicted_log.cpu().numpy().squeeze(-1)
            )

    predicted_log_sales = np.concatenate(predicted_log_values)
    predicted_sales = np.clip(np.expm1(predicted_log_sales), 0, None)

    return {
        "predicted_sales": predicted_sales,
        "predicted_log_sales": predicted_log_sales,
        "predicted_zero": predicted_sales < ZERO_SALES_THRESHOLD,
    }


def predict_zero_aware_model(
    model,
    forecast_data,
    model_forward,
    device,
    eval_batch_size=EVAL_BATCH_SIZE,
):
    model.eval()
    probability_values = []
    positive_log_values = []
    number_of_series = len(forecast_data["x_past"])

    with torch.no_grad():
        for start in range(0, number_of_series, eval_batch_size):
            end = min(start + eval_batch_size, number_of_series)
            batch = build_tensor_batch(forecast_data, start, end, device)
            logits, positive_log = model_forward(model, batch)

            probability_values.append(
                torch.sigmoid(logits).cpu().numpy().squeeze(-1)
            )
            positive_log_values.append(
                positive_log.cpu().numpy().squeeze(-1)
            )

    nonzero_probability = np.concatenate(probability_values)
    positive_log_magnitude = np.concatenate(positive_log_values)
    positive_magnitude = np.clip(np.expm1(positive_log_magnitude), 0, None)
    predicted_sales = nonzero_probability * positive_magnitude

    return {
        "predicted_sales": predicted_sales,
        "predicted_log_sales": np.log1p(predicted_sales),
        "nonzero_probability": nonzero_probability,
        "positive_log_magnitude": positive_log_magnitude,
        "positive_magnitude": positive_magnitude,
        "predicted_zero": (
            nonzero_probability < NONZERO_PROBABILITY_THRESHOLD
        ),
    }


def evaluate_regression_model(model, forecast_data, model_forward, device):
    prediction = predict_regression_model(
        model, forecast_data, model_forward, device
    )
    metrics = calculate_metrics(
        prediction["predicted_sales"],
        forecast_data["actual_sales"],
        prediction["predicted_zero"],
    )
    return metrics, prediction


def evaluate_zero_aware_model(model, forecast_data, model_forward, device):
    prediction = predict_zero_aware_model(
        model, forecast_data, model_forward, device
    )
    metrics = calculate_metrics(
        prediction["predicted_sales"],
        forecast_data["actual_sales"],
        prediction["predicted_zero"],
    )
    return metrics, prediction


def evaluate_baselines(forecast_data):
    return {
        name: calculate_metrics(prediction, forecast_data["actual_sales"])
        for name, prediction in build_baseline_predictions(forecast_data).items()
    }


def create_prediction_table(
    run_id,
    model_name,
    seed,
    subset_name,
    forecast_data,
    prediction,
    series_table,
):
    number_of_series, horizon = forecast_data["actual_sales"].shape
    series_ids = np.repeat(forecast_data["series_ids"], horizon)
    start = forecast_data["forecast_start_idx"]
    time_indices = np.tile(np.arange(start, start + horizon), number_of_series)
    metadata = series_table.set_index("series_idx").loc[
        forecast_data["series_ids"]
    ]

    table = pd.DataFrame({
        "run_id": run_id,
        "model_name": model_name,
        "seed": seed,
        "subset": subset_name,
        "series_idx": series_ids,
        "item_id": np.repeat(metadata["item_id"].to_numpy(), horizon),
        "store_id": np.repeat(metadata["store_id"].to_numpy(), horizon),
        "day": [f"d_{value + 1}" for value in time_indices],
        "date": pd.Timestamp("2011-01-29") + pd.to_timedelta(time_indices, unit="D"),
        "actual_sales": forecast_data["actual_sales"].reshape(-1),
        "predicted_sales": prediction["predicted_sales"].reshape(-1),
        "predicted_log_sales": prediction["predicted_log_sales"].reshape(-1),
        "actual_zero_class": (
            forecast_data["actual_sales"].reshape(-1) == 0
        ).astype("int8"),
        "predicted_zero_class": prediction["predicted_zero"].reshape(-1).astype(
            "int8"
        ),
    })

    if "nonzero_probability" in prediction:
        table["predicted_positive_probability"] = prediction[
            "nonzero_probability"
        ].reshape(-1)
        table["predicted_positive_magnitude"] = prediction[
            "positive_magnitude"
        ].reshape(-1)

    return table


def create_per_series_metrics(
    run_id,
    model_name,
    seed,
    subset_name,
    forecast_data,
    prediction,
    series_table,
):
    actual = forecast_data["actual_sales"]
    predicted = prediction["predicted_sales"]
    absolute_error = np.abs(predicted - actual)
    actual_sum = actual.sum(axis=1)
    metadata = series_table.set_index("series_idx").loc[
        forecast_data["series_ids"]
    ]

    wape = np.divide(
        absolute_error.sum(axis=1),
        actual_sum,
        out=np.full(actual_sum.shape, np.nan),
        where=actual_sum != 0,
    )

    return pd.DataFrame({
        "run_id": run_id,
        "model_name": model_name,
        "seed": seed,
        "subset": subset_name,
        "series_idx": forecast_data["series_ids"],
        "item_id": metadata["item_id"].to_numpy(),
        "store_id": metadata["store_id"].to_numpy(),
        "category_id": metadata["cat_id"].to_numpy(),
        "department_id": metadata["dept_id"].to_numpy(),
        "state_id": metadata["state_id"].to_numpy(),
        "actual_demand_sum": actual_sum,
        "predicted_demand_sum": predicted.sum(axis=1),
        "series_zero_ratio": np.mean(actual == 0, axis=1),
        "mae": np.mean(absolute_error, axis=1),
        "rmse": np.sqrt(np.mean((predicted - actual) ** 2, axis=1)),
        "wape": wape,
        "number_of_actual_zero_days": np.sum(actual == 0, axis=1),
        "number_of_predicted_zero_days": np.sum(
            prediction["predicted_zero"], axis=1
        ),
    })
