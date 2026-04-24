"""ForecasterV2: categorical-embedding encoder + LSTM + optional persistence residual.

Two design fixes over the original LSTMMultiOutput:
  1. Categorical features (bs_id, slice_id, sched, tr) use ``nn.Embedding`` instead
     of going through a StandardScaler. This avoids the val-side ±10^6 explosion
     we saw in trainer_v2's per-client scaler.
  2. For regression tasks, the output adds the last-step value of a chosen
     "persistence feature" so the baseline prediction is built into the graph
     by construction. Dropout / weight decay then can't punish the identity path.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ..data_v2.encoders import FeatureSchema


class ForecasterV2(nn.Module):
    def __init__(
        self,
        schema: FeatureSchema,
        task: Literal["regression", "classification"],
        seq_len: int = 5,
        *,
        cat_embed_dim: int = 8,
        lstm_hidden1: int = 64,
        lstm_hidden2: int = 32,
        fc_hidden: int = 64,
        dropout: float = 0.1,
        persistence_feature: str | None = None,
    ):
        super().__init__()
        self.schema = schema
        self.task = task
        self.seq_len = seq_len
        self.persistence_feature = persistence_feature

        # Validate persistence configuration.
        if persistence_feature is not None:
            if persistence_feature not in schema.continuous:
                raise ValueError(
                    f"persistence_feature={persistence_feature!r} must be in "
                    f"schema.continuous={schema.continuous}"
                )
            self._persistence_idx = schema.continuous.index(persistence_feature)
        else:
            self._persistence_idx = None

        # One embedding table per categorical column.
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(schema.categorical_sizes[col] + 1, cat_embed_dim)
            for col in schema.categorical
        })
        input_dim = cat_embed_dim * schema.n_categorical + schema.n_continuous

        self.lstm1 = nn.LSTM(input_dim, lstm_hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hidden1, lstm_hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden2, fc_hidden)
        self.relu = nn.ReLU(inplace=False)
        self.head = nn.Linear(fc_hidden, 1)

        if task == "regression":
            # Zero-init the head so initial prediction ≈ persistence (when residual is on)
            # or ≈ 0 (when it isn't). Either way, training starts from a sensible prior.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """x_cat: (B, L, n_cat) int64   x_cont: (B, L, n_cont) float32.

        Returns:
          regression → (B, 1) real-valued prediction (+ persistence residual if configured).
          classification → (B, 1) logits (sigmoid applied outside via BCEWithLogitsLoss).
        """
        cats = []
        for i, col in enumerate(self.schema.categorical):
            emb = self.embeddings[col](x_cat[..., i])  # (B, L, embed_dim)
            cats.append(emb)
        x = torch.cat(cats + [x_cont], dim=-1) if cats else x_cont

        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        last = h[:, -1, :]
        h = self.relu(self.fc(self.dropout(last)))
        delta = self.head(h)

        if self.task == "regression" and self._persistence_idx is not None:
            # Residual connection: baseline = last-step value of reference feature.
            baseline = x_cont[:, -1, self._persistence_idx].unsqueeze(-1)
            return baseline + delta
        return delta
