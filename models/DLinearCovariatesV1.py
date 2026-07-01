from types import SimpleNamespace

import torch.nn as nn

from .DLinear import Model as DLinear


class DLinearCovariatesV1(nn.Module):
    """DLinear target forecast with a past-covariate adjustment branch."""

    def __init__(self, seq_len, pred_len, n_covariates):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_covariates = n_covariates

        self.target_dlinear = DLinear(
            SimpleNamespace(
                seq_len=seq_len,
                pred_len=pred_len,
                individual=False,
                enc_in=1,
            )
        )

        self.covariate_head = nn.Linear(seq_len * n_covariates, pred_len)

    def forward(self, x):
        # x shape: [batch, seq_len, channels (log_sales + covariates)]
        # channel 0 is log_sales; remaining channels are covariates.
        target_history = x[:, :, 0:1]
        covariate_history = x[:, :, 1:]

        base_forecast = self.target_dlinear(target_history)

        covariate_flat = covariate_history.reshape(covariate_history.shape[0], -1)
        covariate_adjustment = self.covariate_head(covariate_flat).unsqueeze(-1)

        return base_forecast + covariate_adjustment
