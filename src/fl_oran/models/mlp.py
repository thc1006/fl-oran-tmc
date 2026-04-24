"""MLP used in v1.0.6 (Dense 64 → 32 → 1)."""
from __future__ import annotations

import torch
from torch import nn


class MLPv106(nn.Module):
    """Port of the v1.0.6 Keras model.

    Keras:
      Dense(64, relu, L2=1e-4) → Dropout(0.2) → Dense(32, relu, L2=1e-4) → Dropout(0.2) → Dense(1)
    """

    def __init__(self, in_features: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
