"""Training functions shared by the final model notebooks."""

import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .final_config import (
    CONVERGENCE_MIN_DELTA,
    MAGNITUDE_LOSS_WEIGHT,
    OCCURRENCE_LOSS_WEIGHT,
)
from .final_evaluation import (
    evaluate_regression_model,
    evaluate_zero_aware_model,
)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    return {
        "x_past": batch["x_past"].to(device),
        "x_future": batch["x_future"].to(device),
        "y_log": batch["y_log"].to(device),
        "metadata": batch["metadata"].long().to(device),
    }


def copy_model_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_regression_model(
    create_model,
    train_loader,
    model_forward,
    device,
    learning_rate,
    epochs,
    seed,
    validation_data=None,
    min_delta=CONVERGENCE_MIN_DELTA,
):
    set_random_seed(seed)
    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    best_wape = float("inf")
    best_metrics = None
    best_state = copy_model_state(model)
    best_epoch = 0
    previous_loss = None
    convergence_epoch = None
    history = []

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        train_loader.dataset.set_epoch(epoch - 1)
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            prediction = model_forward(model, batch)
            loss = loss_fn(prediction, batch["y_log"])
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        loss_difference = np.nan

        if previous_loss is not None:
            loss_difference = abs(previous_loss - train_loss)

            if convergence_epoch is None and loss_difference < min_delta:
                convergence_epoch = epoch

        validation_metrics = {}
        checkpoint_saved = False

        if validation_data is not None:
            validation_metrics, _ = evaluate_regression_model(
                model, validation_data, model_forward, device
            )

            if validation_metrics["wape"] < best_wape:
                best_wape = validation_metrics["wape"]
                best_metrics = validation_metrics.copy()
                best_state = copy_model_state(model)
                best_epoch = epoch
                checkpoint_saved = True
        else:
            best_state = copy_model_state(model)
            best_epoch = epoch
            checkpoint_saved = epoch == epochs

        history.append({
            "epoch": epoch,
            "optimizer_updates": epoch * len(train_loader),
            "samples_seen": epoch * len(train_loader.dataset),
            "learning_rate": learning_rate,
            "training_total_loss": train_loss,
            "training_mse": train_loss,
            "training_occurrence_bce": np.nan,
            "training_magnitude_mse": np.nan,
            "training_loss_difference": loss_difference,
            "validation_mae": validation_metrics.get("mae", np.nan),
            "validation_rmse": validation_metrics.get("rmse", np.nan),
            "validation_wape": validation_metrics.get("wape", np.nan),
            "epoch_runtime_seconds": time.perf_counter() - start_time,
            "checkpoint_saved": checkpoint_saved,
        })

        previous_loss = train_loss

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "first_converged_epoch": convergence_epoch,
        "history": history,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def train_zero_aware_model(
    create_model,
    train_loader,
    model_forward,
    device,
    learning_rate,
    epochs,
    seed,
    validation_data=None,
    occurrence_loss_weight=OCCURRENCE_LOSS_WEIGHT,
    magnitude_loss_weight=MAGNITUDE_LOSS_WEIGHT,
    min_delta=CONVERGENCE_MIN_DELTA,
):
    set_random_seed(seed)
    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    occurrence_loss_fn = torch.nn.BCEWithLogitsLoss()
    magnitude_loss_fn = torch.nn.MSELoss()

    best_wape = float("inf")
    best_metrics = None
    best_state = copy_model_state(model)
    best_epoch = 0
    previous_loss = None
    convergence_epoch = None
    history = []

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        train_loader.dataset.set_epoch(epoch - 1)
        model.train()
        total_losses = []
        occurrence_losses = []
        magnitude_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = move_batch_to_device(batch, device)
            y_log = batch["y_log"]
            occurrence_target = (y_log > 0).float()
            positive_mask = occurrence_target.bool()

            optimizer.zero_grad()
            occurrence_logits, positive_log = model_forward(model, batch)

            occurrence_loss = occurrence_loss_fn(
                occurrence_logits, occurrence_target
            )

            if positive_mask.any():
                magnitude_loss = magnitude_loss_fn(
                    positive_log[positive_mask],
                    y_log[positive_mask],
                )
            else:
                magnitude_loss = positive_log.sum() * 0.0

            total_loss = (
                occurrence_loss_weight * occurrence_loss
                + magnitude_loss_weight * magnitude_loss
            )
            total_loss.backward()
            optimizer.step()

            total_losses.append(total_loss.item())
            occurrence_losses.append(occurrence_loss.item())
            magnitude_losses.append(magnitude_loss.item())

        train_loss = float(np.mean(total_losses))
        occurrence_loss = float(np.mean(occurrence_losses))
        magnitude_loss = float(np.mean(magnitude_losses))
        loss_difference = np.nan

        if previous_loss is not None:
            loss_difference = abs(previous_loss - train_loss)

            if convergence_epoch is None and loss_difference < min_delta:
                convergence_epoch = epoch

        validation_metrics = {}
        checkpoint_saved = False

        if validation_data is not None:
            validation_metrics, _ = evaluate_zero_aware_model(
                model, validation_data, model_forward, device
            )

            if validation_metrics["wape"] < best_wape:
                best_wape = validation_metrics["wape"]
                best_metrics = validation_metrics.copy()
                best_state = copy_model_state(model)
                best_epoch = epoch
                checkpoint_saved = True
        else:
            best_state = copy_model_state(model)
            best_epoch = epoch
            checkpoint_saved = epoch == epochs

        history.append({
            "epoch": epoch,
            "optimizer_updates": epoch * len(train_loader),
            "samples_seen": epoch * len(train_loader.dataset),
            "learning_rate": learning_rate,
            "training_total_loss": train_loss,
            "training_mse": np.nan,
            "training_occurrence_bce": occurrence_loss,
            "training_magnitude_mse": magnitude_loss,
            "training_loss_difference": loss_difference,
            "validation_mae": validation_metrics.get("mae", np.nan),
            "validation_rmse": validation_metrics.get("rmse", np.nan),
            "validation_wape": validation_metrics.get("wape", np.nan),
            "epoch_runtime_seconds": time.perf_counter() - start_time,
            "checkpoint_saved": checkpoint_saved,
        })

        previous_loss = train_loss

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "first_converged_epoch": convergence_epoch,
        "history": history,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
