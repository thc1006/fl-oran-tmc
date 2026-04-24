"""FedAvg weight aggregation."""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch


def weighted_average_state_dicts(
    states: Iterable[dict[str, torch.Tensor]],
    weights: Iterable[float],
) -> "OrderedDict[str, torch.Tensor]":
    """Weighted average of state dicts.

    Only floating-point tensors are averaged. Integer tensors (e.g. BatchNorm
    ``num_batches_tracked``) are copied from the first state to avoid dtype
    issues — their exact value is unimportant for FedAvg.
    """
    states = list(states)
    weights = list(weights)
    if not states:
        raise ValueError("states is empty")
    if len(states) != len(weights):
        raise ValueError("states and weights must have same length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("sum of weights must be positive")
    normalized = [w / total for w in weights]

    avg: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, ref in states[0].items():
        if ref.dtype.is_floating_point:
            acc = torch.zeros_like(ref)
            for st, w in zip(states, normalized):
                acc.add_(st[key].to(acc.dtype), alpha=w)
            avg[key] = acc
        else:
            avg[key] = ref.clone()
    return avg
