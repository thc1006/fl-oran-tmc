"""SpikingForecaster: ForecasterV2-shaped wrapper around a spiking state-space backbone.

The backbone (:class:`SpikingSSMBlock`) couples a diagonal selective state-space
recurrence with a per-channel leaky-integrate-and-fire (LIF) output neuron
implemented via :mod:`snntorch`. The continuous SSM output is integrated by
the LIF membrane and emits binary spikes on threshold crossings; the spike
train then becomes the input to the next block (or to the down-projection
into the classifier head).

This file contains the only production code that actually emits spikes — the
energy and accuracy results in the Stage 1 paper come from this implementation.
Per ADR-001 D-20: ``lif_beta=0.9, lif_threshold=1.0, surrogate=atan(alpha=2.0)``,
``T_inner=1`` (one LIF integration per input sequence position).

The class registers two non-persistent buffers (``spike_count`` and
``forward_inferences``) on each block; downstream energy estimation
(``energy_metrics`` module) reads them after a representative forward pass.
``reset_spike_counters()`` zeros both before measurement.
"""
from __future__ import annotations

from typing import Literal

import snntorch as snn
import snntorch.surrogate as sg
import torch
from torch import nn

from ..data_v2.encoders import FeatureSchema


class SpikingSSMBlock(nn.Module):
    """Diagonal selective SSM with a per-channel LIF spiking output.

    For one timestep ``t`` and one channel ``e`` the dynamics are::

        h[b,e,n] <- exp(dt * A[e,n]) * h[b,e,n] + B[e,n] * (in_proj(x))[b,e]
        y[b,e]   = sum_n C[e,n] * h[b,e,n] + D[e] * (in_proj(x))[b,e]
        spk[b,e], mem[b,e] = LIF(y[b,e], mem[b,e])

    where ``A`` is real-negative, log-parameterised; ``B``, ``C`` are learnable
    matrices initialised small; ``D`` is a per-channel skip; and ``LIF`` is
    a stateful leaky-integrate-and-fire neuron with the atan surrogate
    gradient. ``dt`` is fixed (set to ``1.0`` so the discretisation matches
    a unit-time step; learnable per-block dt would be a future extension).

    The spike count buffer accumulates across forward passes (use
    ``reset_spike_counters`` before measurement). Counter updates happen
    in eval mode only — they are detached from the graph so they cannot
    interfere with training.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        lif_threshold: float = 1.0,
        lif_beta: float = 0.9,
        atan_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.lif_threshold = lif_threshold
        self.lif_beta = lif_beta

        # Input projection (in-tree, used as the SSM "B@x" input gate).
        self.in_proj = nn.Linear(d_model, d_model)
        # Diagonal SSM A matrix per channel (stable: A = -exp(A_log)).
        # Initialise with the S4D-Real ramp so all eigenvalues are negative.
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A_init = A_init.expand(d_model, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A_init))
        # B and C are learnable (d_model, d_state), initialised small.
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        # Per-channel skip path.
        self.D = nn.Parameter(torch.ones(d_model))
        # Output projection from the spike train back to d_model.
        self.out_proj = nn.Linear(d_model, d_model)

        # LIF neuron (stateless module; state is passed explicitly).
        self.lif = snn.Leaky(
            beta=lif_beta,
            threshold=lif_threshold,
            spike_grad=sg.atan(alpha=atan_alpha),
        )

        # Energy-metric counters (non-persistent; reset before measurement).
        self.register_buffer("spike_count", torch.tensor(0.0), persistent=False)
        self.register_buffer("forward_inferences", torch.tensor(0.0), persistent=False)

    def reset_spike_counters(self) -> None:
        self.spike_count.zero_()
        self.forward_inferences.zero_()

    def _scan_emit_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SSM scan and return the spike train ``(B, L, d_model)``."""
        b, length, _ = x.shape
        u = self.in_proj(x)                                  # (B, L, d_model)
        A = -torch.exp(self.A_log)                           # (d_model, d_state)
        dA = torch.exp(A)                                    # dt=1 → exp(A)
        h = u.new_zeros(b, self.d_model, self.d_state)
        # snntorch's init_leaky returns 0 (scalar); broadcasting handles shape.
        mem = self.lif.init_leaky()

        spike_outputs: list[torch.Tensor] = []
        for t in range(length):
            # SSM update: h[b,e,n] = dA[e,n] * h[b,e,n] + B[e,n] * u[b,e].
            h = dA.unsqueeze(0) * h + self.B.unsqueeze(0) * u[:, t, :].unsqueeze(-1)
            # SSM output: y[b,e] = sum_n C[e,n] * h[b,e,n] + D[e] * u[b,e].
            y_t = (h * self.C.unsqueeze(0)).sum(dim=-1) + self.D.unsqueeze(0) * u[:, t, :]
            spk_t, mem = self.lif(y_t, mem)
            spike_outputs.append(spk_t)

            # Energy bookkeeping (eval-mode only; detached so no graph effect).
            if not self.training:
                self.spike_count = self.spike_count + spk_t.sum().detach()
                # one inference per row in the batch.
                self.forward_inferences = self.forward_inferences + float(b) / length
        return torch.stack(spike_outputs, dim=1)

    def forward_spikes_only(self, x: torch.Tensor) -> torch.Tensor:
        """Expose the raw spike train for tests + energy measurement."""
        return self._scan_emit_spikes(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, d_model) → (B, L, d_model)`` via spike train + linear projection."""
        return self.out_proj(self._scan_emit_spikes(x))


class SpikingForecaster(nn.Module):
    """Drop-in alternative to ForecasterV2 with a spiking-SSM backbone."""

    def __init__(
        self,
        schema: FeatureSchema,
        task: Literal["regression", "classification"],
        seq_len: int = 5,
        *,
        cat_embed_dim: int = 8,
        backbone_d_model: int = 80,
        backbone_d_state: int = 16,
        n_blocks: int = 2,
        lif_threshold: float = 1.0,
        lif_beta: float = 0.9,
        atan_alpha: float = 2.0,
        fc_hidden: int = 64,
        out_proj_dim: int = 32,
        dropout: float = 0.0,
        persistence_feature: str | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.task = task
        self.seq_len = seq_len
        self.persistence_feature = persistence_feature

        if persistence_feature is not None:
            if persistence_feature not in schema.continuous:
                raise ValueError(
                    f"persistence_feature={persistence_feature!r} must be in "
                    f"schema.continuous={schema.continuous}"
                )
            self._persistence_idx = schema.continuous.index(persistence_feature)
        else:
            self._persistence_idx = None

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(schema.categorical_sizes[col] + 1, cat_embed_dim)
            for col in schema.categorical
        })
        input_dim = cat_embed_dim * schema.n_categorical + schema.n_continuous

        self.in_proj = nn.Linear(input_dim, backbone_d_model)
        self.blocks = nn.ModuleList([
            SpikingSSMBlock(
                d_model=backbone_d_model,
                d_state=backbone_d_state,
                lif_threshold=lif_threshold,
                lif_beta=lif_beta,
                atan_alpha=atan_alpha,
            )
            for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(backbone_d_model, out_proj_dim)

        # Classifier head (no dropout by default per D-20: LIF binarisation
        # already acts as an implicit regulariser).
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_proj_dim, fc_hidden)
        self.relu = nn.ReLU(inplace=False)
        self.head = nn.Linear(fc_hidden, 1)

        if task == "regression":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def reset_spike_counters(self) -> None:
        """Zero all spiking-block counters before measuring energy."""
        for block in self.blocks:
            block.reset_spike_counters()

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """``x_cat: (B, L, n_cat) int64, x_cont: (B, L, n_cont) float32 → (B, 1)``."""
        cats = []
        for i, col in enumerate(self.schema.categorical):
            cats.append(self.embeddings[col](x_cat[..., i]))
        x = torch.cat(cats + [x_cont], dim=-1) if cats else x_cont

        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.out_proj(h)

        last = h[:, -1, :]
        h = self.relu(self.fc(self.dropout(last)))
        delta = self.head(h)

        if self.task == "regression" and self._persistence_idx is not None:
            baseline = x_cont[:, -1, self._persistence_idx].unsqueeze(-1)
            return baseline + delta
        return delta
