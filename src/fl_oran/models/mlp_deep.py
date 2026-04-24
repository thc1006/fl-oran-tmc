"""Enhanced MLP from v1.0.7-2 (Dense 128 → BN → 64 → BN → 32 → 1)."""
from __future__ import annotations

import torch
from torch import nn


class MLPv107_2(nn.Module):
    """Deeper MLP with BatchNorm + Dropout, matching v1.0.7-2."""

    def __init__(
        self,
        in_features: int,
        hidden: tuple[int, ...] = (128, 64, 32),
        dropouts: tuple[float, ...] = (0.3, 0.2, 0.1),
    ):
        super().__init__()
        assert len(hidden) == len(dropouts)
        layers: list[nn.Module] = []
        prev = in_features
        for h, p in zip(hidden, dropouts):
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
