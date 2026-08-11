import torch
import torch.nn as nn
import torch.nn.functional as F

from .DLinearCovariatesV1 import DLinearCovariatesV1


class DLinearFull(nn.Module):
    """DLinear with past covariates, static metadata, and a zero-aware output."""

    def __init__(
        self,
        seq_len,
        pred_len,
        n_past_covariates,
        metadata_cardinalities,
        embedding_dim=4,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_past_covariates = n_past_covariates

        # Forecast demand magnitude from sales and past numerical covariates.
        self.magnitude_model = DLinearCovariatesV1(
            seq_len=seq_len,
            pred_len=pred_len,
            n_covariates=n_past_covariates,
        )

        # Learn one embedding table for each static metadata field.
        self.metadata_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim, padding_idx=0)
            for cardinality in metadata_cardinalities
        ])

        metadata_width = len(metadata_cardinalities) * embedding_dim

        # Metadata provides an additive adjustment to the magnitude forecast.
        self.metadata_magnitude_head = nn.Linear(metadata_width, pred_len)

        # Estimate non-zero demand from all information available at forecast time.
        occurrence_width = (
            seq_len * (1 + n_past_covariates)
            + metadata_width
        )
        self.occurrence_head = nn.Linear(occurrence_width, pred_len)

        # Begin with no metadata adjustment and P(non-zero) = 0.5.
        nn.init.zeros_(self.metadata_magnitude_head.weight)
        nn.init.zeros_(self.metadata_magnitude_head.bias)
        nn.init.zeros_(self.occurrence_head.weight)
        nn.init.zeros_(self.occurrence_head.bias)

    def forward(self, x_past, metadata):
        # x_past: [batch, seq_len, log_sales + past covariates]
        # metadata: [batch, number of metadata fields]
        embedded_metadata = [
            embedding(metadata[:, column_position])
            for column_position, embedding in enumerate(self.metadata_embeddings)
        ]
        metadata_features = torch.cat(embedded_metadata, dim=1)

        raw_log_magnitude = self.magnitude_model(x_past)
        metadata_adjustment = self.metadata_magnitude_head(
            metadata_features
        ).unsqueeze(-1)
        positive_log_magnitude = F.softplus(
            raw_log_magnitude + metadata_adjustment
        )

        occurrence_features = torch.cat(
            [
                torch.flatten(x_past, start_dim=1),
                metadata_features,
            ],
            dim=1,
        )
        occurrence_logits = self.occurrence_head(
            occurrence_features
        ).unsqueeze(-1)

        return occurrence_logits, positive_log_magnitude
