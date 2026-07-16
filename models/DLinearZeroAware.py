import torch
import torch.nn as nn
import torch.nn.functional as F

from .DLinearCovariatesV1 import DLinearCovariatesV1


class DLinearZeroAware(nn.Module):
    """DLinear with separate demand-occurrence and magnitude outputs."""

    def __init__(self, seq_len, pred_len, n_covariates):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_covariates = n_covariates

        # Forecast positive log-demand using the past-covariate DLinear model.
        self.magnitude_model = DLinearCovariatesV1(
            seq_len=seq_len,
            pred_len=pred_len,
            n_covariates=n_covariates,
        )

        # Estimate the probability of non-zero demand for each forecast day
        self.occurrence_head = nn.Linear(
            seq_len * (1 + n_covariates),
            pred_len,
        )

        # Start from a neutral non-zero probability of 0.5
        nn.init.zeros_(self.occurrence_head.weight)
        nn.init.zeros_(self.occurrence_head.bias)

    def forward(self, x):
        # x shape: [batch, seq_len, log_sales + past covariates]
        occurrence_input = torch.flatten(x, start_dim=1)
        occurrence_logits = self.occurrence_head(occurrence_input).unsqueeze(-1)

        raw_log_magnitude = self.magnitude_model(x)
        # https://www.analyticsvidhya.com/blog/2025/12/softplus-activation-function/
        positive_log_magnitude = F.softplus(raw_log_magnitude)

        return occurrence_logits, positive_log_magnitude
