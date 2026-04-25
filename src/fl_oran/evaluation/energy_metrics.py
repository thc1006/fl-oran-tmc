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

**Caveat for ``decode_mode='sum'`` cells (T_inner > 1)**: in sum mode the
per-block ``out_proj`` consumes a real-valued (rate-coded) tensor in
``[0, 1]``, not a binary spike train. The "subtract dense MAC + add
spike-driven AC" accounting below treats those operations as if their
input were still binary, which under-estimates the sparsity-aware energy
of those cells. The headline preregistered Spiking variant uses
``t_inner=1`` (where all decode modes collapse and the assumption holds);
the audit ``spiking_t5sum`` variants are reported with the same formula
for consistency, and the GO/NO-GO decision is robust to the bias because
all three hardware accountings agree on the trade-off classification for
those cells. A Stage 2 follow-up would refine the formula by detecting
``decode_mode='sum'`` blocks and skipping their MAC subtraction.
"""
from __future__ import annotations

from typing import Any

import torch

from ..models.spiking_forecaster import SpikingSSMBlock

PJ_PER_MAC_FP32: float = 4.6
PJ_PER_AC_FP32: float = 0.9


def count_flops_total(model: torch.nn.Module, x_cat: torch.Tensor, x_cont: torch.Tensor) -> float:
    """Per-inference dense MAC count for ``model.forward(x_cat, x_cont)``.

    Combines:
    * ``fvcore.nn.FlopCountAnalysis`` for traceable ops (Linear, Conv, matmul, ...)
    * Hand-counted MACs for ``nn.LSTM`` and ``nn.GRU`` modules, which fvcore
      does not trace into (it sees only the C++ kernel boundary). Without
      this correction, an LSTM-based model reports ~80× too few FLOPs and
      its energy estimate is severely undercounted.
    * **Subtracts** the post-spike ``out_proj`` operations of every
      :class:`SpikingSSMBlock` so they are not double-counted as dense
      MACs in this number — see :func:`count_post_spike_mac_to_remove`.

    .. note::

       Side effect: this function calls ``model.eval()`` to drive
       ``fvcore`` along the inference code path. The caller's training
       mode is therefore overwritten. ``estimate_energy_pJ_per_inference``
       handles this internally; standalone callers who care about preserving
       train mode should re-invoke ``model.train()`` after this returns.
    """
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
    fvcore_flops = float(analysis.total())

    # Hand-count MACs for recurrent modules fvcore doesn't trace.
    seq_len = int(x_cat.shape[1])
    rnn_macs_per_inference = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
            input_size = module.input_size
            hidden_size = module.hidden_size
            num_layers = module.num_layers
            n_gates = 4 if isinstance(module, torch.nn.LSTM) else 3
            # Per timestep, per layer: n_gates × (I + H + 1) × H MACs.
            # Layers after the first take hidden_size as input.
            macs_per_step = (
                n_gates * (input_size + hidden_size + 1) * hidden_size
                + (num_layers - 1) * n_gates * (2 * hidden_size + 1) * hidden_size
            )
            rnn_macs_per_inference += macs_per_step * seq_len

    rnn_macs_total = rnn_macs_per_inference * int(n_inferences)

    # Remove post-spike out_proj MACs from the dense count: those operations
    # consume a binary spike train as input and are accumulate-only (AC), not
    # multiply-accumulate (MAC). They are added to the sops total instead via
    # :func:`count_block_sops`.
    post_spike_macs_per_inference = count_post_spike_mac_to_remove(model, seq_len)
    post_spike_macs_total = post_spike_macs_per_inference * int(n_inferences)

    return (fvcore_flops + rnn_macs_total - post_spike_macs_total) / n_inferences


def count_post_spike_mac_to_remove(model: torch.nn.Module, seq_len: int) -> int:
    """MAC count attributable to ``SpikingSSMBlock.out_proj`` per inference.

    These MACs are removed from the dense-FLOPs total in
    :func:`count_flops_total` because their input is a binary spike train,
    making the multiplications degenerate (1×w or 0×w). The corresponding
    accumulate operations are reported as sops by :func:`count_block_sops`
    so the energy formula is not double-counting.
    """
    total = 0
    for module in model.modules():
        if isinstance(module, SpikingSSMBlock):
            out = module.out_proj
            in_features = out.in_features
            out_features = out.out_features
            # Linear(in_features → out_features) over seq_len timesteps:
            # in_features × out_features MACs per timestep.
            total += in_features * out_features * seq_len
    return total


def count_block_sops(model: torch.nn.Module) -> float:
    """Per-inference **spike-driven** AC count across all SpikingSSMBlock instances.

    Each emitted LIF spike triggers ``out_proj.out_features`` accumulate
    operations on the post-spike ``out_proj`` Linear weights, so the
    spike-driven AC count for one block over one inference equals
    ``spike_count_block × out_proj.out_features`` divided by the number
    of inferences observed.

    The structural term (the dense ``in_features × out_features × seq_len``
    that ``out_proj`` would cost if its input were dense rather than a spike
    train) is **not** added here. It is netted out of the dense MAC total in
    :func:`count_flops_total` via :func:`count_post_spike_mac_to_remove`, so
    the energy formula

        total_sparsity = (flops_full_dense - structural) * pJ_per_MAC + sops_spike * pJ_per_AC

    is consistent: the dense-MAC term subtracts the cost we are not paying,
    and this function adds back only the cost we ARE paying (one AC per
    actual 1-spike event × the downstream fan-out).

    Returns 0.0 if no SpikingSSMBlock submodules exist (LSTM and Mamba).
    Requires that the model was previously run in eval mode without an
    intervening :meth:`reset_spike_counters` so the per-block spike buffers
    contain the data to read.
    """
    sops_total = 0.0
    inferences_max = 0.0
    for module in model.modules():
        if isinstance(module, SpikingSSMBlock):
            out_f = float(module.out_proj.out_features)
            # Each emitted spike causes `out_features` accumulates downstream;
            # spike_count is the cumulative count of 1-valued spike events.
            sops_total += float(module.spike_count) * out_f
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

    Reports **three energy values** corresponding to three deployment-target
    accounting models, plus the dense-MAC and sparse-AC counts that go into
    each:

    * ``total_energy_pJ_gpu_dense`` — worst case for spiking. Every Linear
      / Conv operation costs a MAC regardless of input value (i.e., a
      standard GPU/CPU matmul that does not exploit input sparsity).
      ``= (flops_dense_full + rnn_macs) * 4.6 pJ``.
    * ``total_energy_pJ_sparsity_aware`` — sparsity-aware accelerator
      that detects 0-spike inputs and skips the multiplication. The
      post-spike ``out_proj`` of every ``SpikingSSMBlock`` is treated as
      AC for actual spike events, MAC for dense events. This is the
      "headline" number reported in §6.4 of the paper.
    * ``total_energy_pJ_neuromorphic`` — idealised neuromorphic chip
      where **all** post-spike Linear operations downstream of any LIF
      neuron are AC. For the current SpikingForecaster only the
      per-block ``out_proj`` directly receives spikes (the classifier
      head receives a dense float vector after out_proj), so for our
      architecture ``neuromorphic == sparsity_aware``. The placeholder
      is exposed so future models with truly spike-stacked layers can
      report the deeper savings.

    LSTM and Mamba models have no spiking blocks and identical numbers
    in all three columns.
    """
    if hasattr(model, "reset_spike_counters"):
        model.reset_spike_counters()

    flops_post_subtraction = count_flops_total(model, x_cat, x_cont)

    # Trigger spike accumulation if this is a Spiking model.
    if hasattr(model, "reset_spike_counters"):
        model.reset_spike_counters()
        model.eval()
        with torch.no_grad():
            _ = model(x_cat, x_cont)

    sops = count_block_sops(model)

    # Reconstruct the worst-case dense-MAC count by adding back the
    # post-spike out_proj structural MACs that count_flops_total subtracted.
    seq_len = int(x_cat.shape[1])
    structural = float(count_post_spike_mac_to_remove(model, seq_len))
    flops_dense_full = flops_post_subtraction + structural

    total_gpu = flops_dense_full * PJ_PER_MAC_FP32
    total_sparsity = flops_post_subtraction * PJ_PER_MAC_FP32 + sops * PJ_PER_AC_FP32
    # For SpikingForecaster as currently structured the only spike-input
    # Linear is each block's out_proj; the classifier head receives a dense
    # float vector after out_proj. Hence neuromorphic == sparsity_aware
    # for this architecture. The two columns are reported separately so
    # downstream tooling does not have to special-case "this model has no
    # deeper spike stack" — when a future architecture stacks spike-input
    # layers (e.g. spike-input fc), this branch will diverge.
    total_neuromorphic = total_sparsity

    return {
        # Backwards-compatible legacy keys (== sparsity_aware accounting):
        "flops": flops_post_subtraction,
        "sops": sops,
        "total_energy_pJ": total_sparsity,
        "pj_per_mac": PJ_PER_MAC_FP32,
        "pj_per_ac": PJ_PER_AC_FP32,
        # Three-hardware accounting:
        "flops_dense_full": flops_dense_full,
        "structural_post_spike_mac": structural,
        "total_energy_pJ_gpu_dense": total_gpu,
        "total_energy_pJ_sparsity_aware": total_sparsity,
        "total_energy_pJ_neuromorphic": total_neuromorphic,
    }
