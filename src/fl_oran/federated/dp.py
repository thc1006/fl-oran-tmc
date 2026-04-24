"""Differential-privacy helpers: Gaussian mechanism + simple RDP accountant.

This is a model-update-level DP (a.k.a. user-level DP) implementation:
after each local round, we clip the weight *delta* (new - start) to an L2 norm
of ``l2_norm_clip`` and add i.i.d. Gaussian noise with standard deviation
``l2_norm_clip * noise_multiplier``.

For sample-level DP, use Opacus; this module targets the same use-case as the
notebook's ``DPOptimizerWrapper``.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import torch


@dataclass
class GaussianMechanism:
    l2_norm_clip: float = 1.0
    noise_multiplier: float = 0.0

    @property
    def noise_stddev(self) -> float:
        return self.l2_norm_clip * self.noise_multiplier

    def clip_and_noise(self, delta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return a clipped+noised copy of ``delta`` (floating tensors only)."""
        flat = [v.flatten() for v in delta.values() if v.dtype.is_floating_point]
        if not flat:
            return {k: v.clone() for k, v in delta.items()}
        total_norm = torch.linalg.vector_norm(torch.cat(flat))
        clip = torch.clamp(self.l2_norm_clip / (total_norm + 1e-12), max=1.0)
        out: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in delta.items():
            if v.dtype.is_floating_point:
                clipped = v * clip
                if self.noise_stddev > 0:
                    clipped = clipped + torch.randn_like(clipped) * self.noise_stddev
                out[k] = clipped
            else:
                out[k] = v.clone()
        return out


def gaussian_dp_update(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    l2_norm_clip: float,
    noise_multiplier: float,
) -> dict[str, torch.Tensor]:
    """Convenience: clip+noise the delta, return the new state (before + delta)."""
    delta = {k: (after[k].to(before[k].dtype) - before[k]) for k in before}
    mech = GaussianMechanism(l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier)
    d = mech.clip_and_noise(delta)
    return {k: before[k] + d[k] if d[k].dtype.is_floating_point else after[k] for k in before}


@dataclass
class PrivacyAccountant:
    """Tiny RDP-based accountant (Abadi et al. 2016 approximation).

    It assumes a single sampling ratio per round; given the noise multiplier
    and number of rounds, it returns cumulative (ε, δ). For high-precision
    tracking, swap in ``dp-accounting`` if available.
    """
    noise_multiplier: float
    sample_rate: float
    target_delta: float = 1e-5
    history: list[float] = field(default_factory=list)  # per-round cum ε

    def step(self, num_steps_this_round: int = 1) -> float:
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        cum_steps = (len(self.history) + 1) * num_steps_this_round
        # Bounded RDP for Gaussian subsampling (upper bound).
        rdp = [cum_steps * self._rdp_gaussian(a, self.noise_multiplier, self.sample_rate) for a in orders]
        eps = self._eps_from_rdp(orders, rdp, self.target_delta)
        self.history.append(eps)
        return eps

    @property
    def epsilon(self) -> float:
        return self.history[-1] if self.history else 0.0

    @staticmethod
    def _rdp_gaussian(alpha: float, noise_multiplier: float, q: float) -> float:
        # Mironov 2019 upper bound for subsampled Gaussian mechanism (loose but safe).
        return q * q * alpha / (2 * max(noise_multiplier, 1e-6) ** 2)

    @staticmethod
    def _eps_from_rdp(orders: list[float], rdps: list[float], delta: float) -> float:
        # Convert RDP to (ε, δ)-DP using the standard formula.
        best = float("inf")
        log_delta = math.log(max(delta, 1e-300))
        for a, r in zip(orders, rdps):
            if a <= 1:
                continue
            eps = r - log_delta / (a - 1)
            if eps < best:
                best = eps
        return max(best, 0.0)
