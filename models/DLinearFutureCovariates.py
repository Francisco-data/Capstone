from types import SimpleNamespace

import torch
import torch.nn as nn

from .DLinear import Model as DLinear


class DLinearFutureCovariates(nn.Module):
    """DLinear forecast adjusted by past and future-known covariates."""

    def __init__(
        self,
        seq_len,
        pred_len,
        n_past_covariates,
        n_future_covariates,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_past_covariates = n_past_covariates
        self.n_future_covariates = n_future_covariates

        # Original DLinear branch uses only the historical target
        self.target_dlinear = DLinear(
            SimpleNamespace(
                seq_len=seq_len,
                pred_len=pred_len,
                individual=False,
                enc_in=1,
            )
        )

        # These branches learn adjustments from past and future covariates
        self.past_covariate_head = nn.Linear(
            seq_len * n_past_covariates,
            pred_len,
        )
        self.future_covariate_head = nn.Linear(
            pred_len * n_future_covariates,
            pred_len,
        )

    def forward(self, x_past, x_future):
        # x_past: [batch, seq_len, log_sales + past covariates]
        # x_future: [batch, pred_len, future-known covariates]
        target_history = x_past[:, :, 0:1]
        past_covariates = x_past[:, :, 1:]

        base_forecast = self.target_dlinear(target_history)

        past_flat = torch.flatten(past_covariates, start_dim=1)
        past_adjustment = self.past_covariate_head(past_flat).unsqueeze(-1)

        future_flat = torch.flatten(x_future, start_dim=1)
        future_adjustment = self.future_covariate_head(future_flat).unsqueeze(-1)

        return base_forecast + past_adjustment + future_adjustment
