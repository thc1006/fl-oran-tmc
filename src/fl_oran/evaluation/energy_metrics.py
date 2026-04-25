"""Energy-estimate utilities used by the Stage 1 paper (ADR-001 D-20).

Two terms are tracked per inference:

* ``flops`` — total floating-point multiply-accumulate operations measured by
  :class:`fvcore.nn.FlopCountAnalysis` over the entire model. Includes the
  encoder, classifier head, and any dense projections inside the backbone
  blocks (Mamba's ``in_proj``, ``x_proj``, ``out_proj``; SpikingSSMBlock's
  ``in_proj``, scan, ``out_proj``). Each MAC is priced at
  :data:`PJ_PER_MAC_FP32` = 4.6 pJ (Horowitz, ISSCC 2014, 45nm CMOS).

* ``sops`` — synaptic operations: cumulative LIF spikes weighted by the
  fan-out of the layer that consumes them. For each
  :class:`fl_oran.models.spiking_forecaster.SpikingSSMBlock`, downstream
  consumer = ``out_proj`` of width ``d_model``, so
  ``sops_block = spike_count_block * d_model``. Each AC is priced at
  :data:`PJ_PER_AC_FP32` = 0.9 pJ (same source).

The reported ``total_energy_pJ`` is an *upper bound*: the dense FLOPs term
double-counts the post-spike out_proj for SpikingForecaster (those ops are
truly accumulate-only when their input is binary, but fvcore reports them as
MACs). The paper's §5 limitations section acknowledges this and reports
``backbone_only_energy_ratio`` separately for transparency.
"""
from __future__ import annotations

from typing import Any

import torch

from ..models.spiking_forecaster import SpikingSSMBlock

PJ_PER_MAC_FP32: float = 4.6
PJ_PER_AC_FP32: float = 0.9


def count_flops_total(model: torch.nn.Module, x_cat: torch.Tensor, x_cont: torch.Tensor) -> float:
    """Per-inference dense MAC count for ``model.forward(x_cat, x_cont)``."""
    from fvcore.nn import FlopCountAnalysis

    n_inferences = float(x_cat.shape[0])
    if n_inferences == 0:
        return 0.0

    model.eval()
    analysis = FlopCountAnalysis(model, (x_cat, x_cont))
    # fvcore loudly warns about untracked snntorch.Leaky / custom ops; silence them
    # for the metric path. They are not MAC contributors anyway.
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    return float(analysis.total()) / n_inferences


def count_block_sops(model: torch.nn.Module) -> float:
    """Per-inference synaptic-op count summed across all SpikingSSMBlock instances.

    Returns 0.0 if no SpikingSSMBlock submodules exist (LSTM and Mamba models).
    Requires that the model was previously run in eval mode without an
    intervening :meth:`reset_spike_counters` so the per-block spike buffers
    contain the data to read.
    """
    sops_total = 0.0
    inferences_max = 0.0
    for module in model.modules():
        if isinstance(module, SpikingSSMBlock):
            fan_out = float(module.out_proj.out_features)
            sops_total += float(module.spike_count) * fan_out
            inferences_max = max(inferences_max, float(module.forward_inferences))
    if inferences_max == 0.0:
        return 0.0
    return sops_total / inferences_max


def estimate_energy_pJ_per_inference(
    model: torch.nn.Module,
    x_cat: torch.Tensor,
    x_cont: torch.Tensor,
) -> dict[str, Any]:
    """Run a measurement forward pass and return per-inference energy stats.

    Resets spike counters first, runs the model in eval mode under
    ``torch.no_grad()``, then computes both terms. Flops are read from
    fvcore (which itself runs a separate forward pass internally).
    """
    if hasattr(model, "reset_spike_counters"):
        model.reset_spike_counters()

    flops = count_flops_total(model, x_cat, x_cont)

    # Trigger spike accumulation if this is a Spiking model.
    if hasattr(model, "reset_spike_counters"):
        model.reset_spike_counters()
        model.eval()
        with torch.no_grad():
            _ = model(x_cat, x_cont)

    sops = count_block_sops(model)
    total = flops * PJ_PER_MAC_FP32 + sops * PJ_PER_AC_FP32
    return {
        "flops": flops,
        "sops": sops,
        "total_energy_pJ": total,
        "pj_per_mac": PJ_PER_MAC_FP32,
        "pj_per_ac": PJ_PER_AC_FP32,
    }
