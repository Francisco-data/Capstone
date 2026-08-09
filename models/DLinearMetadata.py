import torch
import torch.nn as nn

from .DLinearCovariatesV1 import DLinearCovariatesV1


class DLinearMetadata(nn.Module):
    """Past-covariate DLinear with an adjustment from static metadata."""

    def __init__(
        self,
        seq_len,
        pred_len,
        n_covariates,
        metadata_cardinalities,
        embedding_dim=4,
    ):
        super().__init__()

        self.base_model = DLinearCovariatesV1(
            seq_len=seq_len,
            pred_len=pred_len,
            n_covariates=n_covariates,
        )

        # One embedding table is learned for each categorical metadata column
        self.metadata_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim, padding_idx=0)
            for cardinality in metadata_cardinalities
        ])

        # # features * embedding dimension
        metadata_width = len(metadata_cardinalities) * embedding_dim
        # Linear layer
        self.metadata_head = nn.Linear(metadata_width, pred_len)

        # Begin with no metadata adjustment, then learn it during training
        nn.init.zeros_(self.metadata_head.weight)
        nn.init.zeros_(self.metadata_head.bias)

    def forward(self, x, metadata):
        # x contains past log_sales and numerical covariates
        base_forecast = self.base_model(x)

        # metadata shape: [batch, number of metadata columns]
        embedded_metadata = [
            embedding(metadata[:, column_position])
            for column_position, embedding in enumerate(self.metadata_embeddings)
        ]

        # Concatenate embeddings
        metadata_features = torch.cat(embedded_metadata, dim=1)
        metadata_adjustment = self.metadata_head(metadata_features).unsqueeze(-1)

        return base_forecast + metadata_adjustment
