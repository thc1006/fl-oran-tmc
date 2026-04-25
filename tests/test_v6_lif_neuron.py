"""Sanity tests for the LIF neuron + atan surrogate gradient used by SpikingSSMBlock.

These tests pin the third-party `snntorch.Leaky` semantics that our
`SpikingSSMBlock` will rely on. If a future snntorch upgrade changes the
behaviour (e.g. spike convention, surrogate-gradient formula), these
tests fail loudly so we know to re-derive our energy and accuracy
metrics rather than silently train against a different model.

Per ADR-001 D-20: LIF parameters used in Stage 1 are
`beta=0.9, threshold=1.0, surrogate=atan(alpha=2.0)`.
"""
from __future__ import annotations

import math

import pytest
import torch

snn = pytest.importorskip("snntorch")
sg = pytest.importorskip("snntorch.surrogate")


def test_lif_emits_binary_spikes():
    """Spikes from snntorch.Leaky are ∈ {0, 1} regardless of input scale."""
    lif = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=sg.atan(alpha=2.0))
    mem = lif.init_leaky()
    # Mix of below- and above-threshold inputs.
    x = torch.tensor([[-5.0, 0.0, 0.5, 1.0, 5.0]])
    spk, _ = lif(x, mem)
    unique = torch.unique(spk).tolist()
    assert set(unique).issubset({0.0, 1.0}), f"non-binary spikes: {unique}"


def test_lif_membrane_integration_matches_euler():
    """Hand-rolled membrane update matches snntorch.Leaky for one step.

    snntorch's Leaky neuron uses the discrete-time update
        u_t = beta * u_{t-1} + I_t,
        spike_t = 1 if u_t >= threshold else 0,
        u_t -= threshold * spike_t  (subtract-on-spike reset).
    """
    beta, threshold = 0.9, 1.0
    lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg.atan(alpha=2.0))

    # Drive a single neuron with an input sequence that crosses threshold.
    inputs = torch.tensor([[0.4], [0.4], [0.4], [0.4]])  # cumulative ~1.36 over 4 steps
    mem = lif.init_leaky()

    # Manual reference with the same rule.
    ref_mem = torch.zeros(1, 1)
    spk_history = []
    ref_spk_history = []
    for t in range(4):
        spk_t, mem = lif(inputs[t : t + 1], mem)
        spk_history.append(float(spk_t.item()))
        # Hand-roll: pre-spike membrane.
        ref_mem = beta * ref_mem + inputs[t : t + 1]
        ref_spike = (ref_mem >= threshold).float()
        ref_mem = ref_mem - threshold * ref_spike
        ref_spk_history.append(float(ref_spike.item()))

    assert spk_history == ref_spk_history, (
        f"snntorch spike history {spk_history} != hand-rolled {ref_spk_history}"
    )


def test_atan_surrogate_derivative_at_threshold():
    """atan surrogate spike gradient at the threshold equals the closed-form value.

    snntorch's atan surrogate uses
        d spike / d u = alpha / (2 * (1 + (pi * alpha / 2 * (u - threshold))**2))
    so at u == threshold the value is alpha / 2.

    ADR D-20 picks alpha=2 → derivative at threshold = 1.0.
    """
    alpha = 2.0
    threshold = 1.0
    surrogate = sg.atan(alpha=alpha)

    u = torch.tensor([threshold], requires_grad=True)
    # Apply surrogate spike function.
    spk = surrogate(u - threshold)
    spk.backward()
    assert u.grad is not None
    assert math.isclose(u.grad.item(), alpha / 2.0, abs_tol=1e-6), u.grad.item()


def test_lif_gradient_flows_through_surrogate():
    """End-to-end: gradient on a downstream loss reaches the LIF input."""
    lif = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=sg.atan(alpha=2.0))
    mem = lif.init_leaky()
    x = torch.randn(8, 4, requires_grad=True)
    spk, _ = lif(x, mem)
    loss = spk.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.norm().item() > 0, "no gradient flowed back through surrogate"
