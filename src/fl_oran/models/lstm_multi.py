"""LSTM multi-output model from v1.0.7-1.

Keras original:
    LSTM(64, return_sequences=True)
    LSTM(32, return_sequences=False)
    Dense(64, relu)
    Dense(n_targets, sigmoid)   # combined head

For clarity and proper loss handling, we use a combined head that emits one
logit per target; sigmoid + BCE for classification targets and MSE for
regression targets are applied in the training loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MultiOutputSpec:
    regression_targets: list[str]
    classification_targets: list[str]

    @property
    def all_targets(self) -> list[str]:
        return list(self.regression_targets) + list(self.classification_targets)

    @property
    def n_total(self) -> int:
        return len(self.all_targets)


class LSTMMultiOutput(nn.Module):
    def __init__(
        self,
        in_features: int,
        sequence_length: int,
        output_spec: MultiOutputSpec,
        hidden1: int = 64,
        hidden2: int = 32,
        fc_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.sequence_length = sequence_length
        self.output_spec = output_spec

        self.lstm1 = nn.LSTM(in_features, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, fc_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(fc_hidden, output_spec.n_total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> logits: (B, n_targets). Sigmoid is applied in loss."""
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        last = out[:, -1, :]  # many-to-one
        h = self.relu(self.fc(self.dropout(last)))
        return self.head(h)
