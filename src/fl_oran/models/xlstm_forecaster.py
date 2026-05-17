"""xLSTMForecaster: ForecasterV2-shaped wrapper around xLSTM-sLSTM backbone.

Mirrors :class:`fl_oran.models.forecaster_v2.ForecasterV2` exactly except that
the temporal trunk is replaced by N stacked sLSTM cells.

xLSTM is the 2024 NeurIPS extension of LSTM by Beck et al. (arXiv:2405.04517).
This implementation includes ONLY the sLSTM (scalar memory) variant; mLSTM
(matrix memory) is deferred — at our seq_len=5 the matrix memory's
associative-recall advantage does not materialize, and its extra parameter
cost would push us beyond the ~40-50K parity budget.

Key differences from classical LSTM (1997):

1. **Exponential input gate**: ``i_t = exp(ĩ_t)`` instead of sigmoid. This lets
   the model dramatically up-weight specific tokens (e.g., revising stored
   decisions) — a limitation Beck et al. identify in classical LSTM
   (their Fig. 2 Nearest-Neighbor-Search benchmark).

2. **Normalizer state**: ``n_t = f_t · n_{t-1} + i_t`` accumulates gate
   magnitude. Hidden state is normalized: ``h_t = o_t · (c_t / max(|n_t|, 1))``.
   This compensates for the unbounded exponential input gate.

3. **Stabilizer state** (paper eq 15-17): ``m_t = max(log(f_t) + m_{t-1},
   log(i_t))``. Used to compute stabilized gates ``i'_t = exp(ĩ_t - m_t)``
   and ``f'_t = exp(log(f_t) + m_{t-1} - m_t)`` that are mathematically
   equivalent to the unstabilized version but numerically safe under bf16.

We use **sigmoid forget gate** (Beck et al. show both sigmoid and exp work;
sigmoid is the safer default for our short seq_len=5 setting). Multi-head
sLSTM is also deferred — single head with single memory cell per layer is
sufficient at our scale.

Reference impl: https://github.com/NX-AI/xlstm
Time-series adaptation precedent: xLSTMTime (Alharthi & Mahmood,
arXiv:2407.10240, preprint only) — their LTSF tweaks are designed for
long sequences; we use the vanilla Beck 2024 sLSTM formulation.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..data_v2.encoders import FeatureSchema


class sLSTMCell(nn.Module):
    """Single-head, single-cell xLSTM-sLSTM step (paper eq 8-17).

    Forward signature matches the standard recurrent-cell convention:

        h_new, state_new = cell(x_t, state_prev)

    where ``state = (h, c, n, m)`` are the four persistent buffers per step.

    Args:
        input_size: dimensionality of x_t input vector.
        hidden_size: dimensionality of h_t hidden / c_t cell / n_t normalizer /
            m_t stabilizer (all share the same hidden_size).
        forget_gate: ``"sigmoid"`` (default, safer) or ``"exp"`` (paper allows
            both; exp is required for parity tasks but more numerically
            volatile). We use sigmoid by default.
    """

    def __init__(self, input_size: int, hidden_size: int, *,
                 forget_gate: str = "sigmoid") -> None:
        super().__init__()
        if forget_gate not in ("sigmoid", "exp"):
            raise ValueError(
                f"forget_gate must be 'sigmoid' or 'exp'; got {forget_gate!r}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate = forget_gate

        # Input projections W · x_t (paper eq 11-14). Bias term included.
        self.w_z = nn.Linear(input_size, hidden_size, bias=True)
        self.w_i = nn.Linear(input_size, hidden_size, bias=True)
        self.w_f = nn.Linear(input_size, hidden_size, bias=True)
        self.w_o = nn.Linear(input_size, hidden_size, bias=True)

        # Recurrent projections R · h_{t-1} (paper eq 11-14, no bias — biased
        # via the corresponding W projection).
        self.r_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_state(self, batch_size: int, device: torch.device,
                   dtype: torch.dtype = torch.float32):
        """Return zero state ``(h, c, n, m)`` of shape ``(B, hidden_size)``.

        All four states init to zero per paper eq 8-9 recurrence convention
        (``n_t = f_t · n_{t-1} + i_t`` with ``n_0 = 0``). Distinct tensors are
        allocated for each state to avoid aliasing surprises under
        ``torch.compile`` and to make any future in-place mutation safe.
        """
        kw = {"device": device, "dtype": dtype}
        h0 = torch.zeros(batch_size, self.hidden_size, **kw)
        c0 = torch.zeros(batch_size, self.hidden_size, **kw)
        n0 = torch.zeros(batch_size, self.hidden_size, **kw)
        m0 = torch.zeros(batch_size, self.hidden_size, **kw)
        return h0, c0, n0, m0

    def forward(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """One time step. x_t: (B, input_size). state: (h, c, n, m) each (B, hidden_size).

        Returns:
            h_new: (B, hidden_size) hidden state.
            new_state: (h_new, c_new, n_new, m_new).
        """
        h_prev, c_prev, n_prev, m_prev = state

        # Pre-activations (paper eq 11-14 LHS)
        z_tilde = self.w_z(x_t) + self.r_z(h_prev)        # (B, H)
        i_tilde = self.w_i(x_t) + self.r_i(h_prev)        # (B, H)
        f_tilde = self.w_f(x_t) + self.r_f(h_prev)        # (B, H)
        o_tilde = self.w_o(x_t) + self.r_o(h_prev)        # (B, H)

        # Stabilizer state (paper eq 15). log(i_t) = ĩ_t (since i = exp(ĩ)).
        # log(f_t) depends on forget-gate variant.
        log_i = i_tilde
        if self.forget_gate == "sigmoid":
            # log(sigmoid(f̃)) — use logsigmoid for numerical stability
            log_f = F.logsigmoid(f_tilde)
        else:  # "exp"
            log_f = f_tilde

        # Stabilizer update: m_t = max(log(f) + m_{t-1}, log(i))
        m_new = torch.maximum(log_f + m_prev, log_i)

        # Stabilized gates (paper eq 16-17). exp(x - m) is bounded ≤ 1.
        i_prime = torch.exp(log_i - m_new)
        f_prime = torch.exp(log_f + m_prev - m_new)

        # Cell input (paper eq 11). tanh activation.
        z = torch.tanh(z_tilde)

        # Cell + normalizer updates (paper eq 8-9, with stabilized gates).
        c_new = f_prime * c_prev + i_prime * z
        n_new = f_prime * n_prev + i_prime

        # Output gate (paper eq 14, sigmoid as in classical LSTM).
        o = torch.sigmoid(o_tilde)

        # Hidden state with stabilized normalization. Paper eq 10 (unstabilized)
        # writes ``h = o · c / max(|n|, 1)``. Under stabilization we carry
        # ``c'_t = c_t · exp(-m_t)`` and ``n'_t = n_t · exp(-m_t)``, so the
        # mathematically-equivalent clamp threshold becomes ``exp(-m_t)``,
        # NOT 1. Derivation:
        #
        #   h_t  = o · c_t / max(|n_t|, 1)
        #        = o · (c'_t · exp(m_t)) / max(|n'_t| · exp(m_t), 1)
        #        = o · c'_t / max(|n'_t|, exp(-m_t))                    .
        #
        # When ``m_t`` is very negative (signal collapsed), ``exp(-m_t)``
        # may overflow to inf; the resulting ``c'/inf = 0`` is the
        # numerically correct behavior (it matches the unstabilized output
        # ``o · c · exp(m_t) ≈ 0`` for very negative m). Backward through
        # ``1/inf`` is 0, so no NaN gradient.
        n_safe = torch.maximum(n_new.abs(), torch.exp(-m_new))
        h_new = o * (c_new / n_safe)

        return h_new, (h_new, c_new, n_new, m_new)


class xLSTMForecaster(nn.Module):
    """Drop-in alternative to ForecasterV2 with stacked sLSTM cells.

    Encoder (categorical embeddings + continuous concat) and head
    (``fc → relu → head``) are identical to ForecasterV2 so that any
    AUC / loss difference is attributable to the temporal trunk only.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        task: Literal["regression", "classification"],
        seq_len: int = 5,
        *,
        cat_embed_dim: int = 8,
        hidden_size: int = 48,
        n_layers: int = 2,
        fc_hidden: int = 64,
        dropout: float = 0.1,
        forget_gate: str = "sigmoid",
        persistence_feature: str | None = None,
        drop_categorical: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.task = task
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Categorical-drop logic mirrors ForecasterV2 (R2 C4 ablation surface).
        self.drop_categorical: tuple[str, ...] = tuple(drop_categorical or ())
        for col in self.drop_categorical:
            if col not in schema.categorical:
                raise ValueError(
                    f"drop_categorical={col!r} not in schema.categorical"
                    f"={schema.categorical}"
                )
        self._kept_cat_names = tuple(
            col for col in schema.categorical if col not in self.drop_categorical
        )
        self._kept_cat_indices = tuple(
            i for i, col in enumerate(schema.categorical)
            if col not in self.drop_categorical
        )

        if persistence_feature is not None:
            if persistence_feature not in schema.continuous:
                raise ValueError(
                    f"persistence_feature={persistence_feature!r} must be in "
                    f"schema.continuous={schema.continuous}"
                )
            self._persistence_idx = schema.continuous.index(persistence_feature)
        else:
            self._persistence_idx = None
        self.persistence_feature = persistence_feature

        # Categorical embeddings (one per kept column, +1 row for OOV).
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(schema.categorical_sizes[col] + 1, cat_embed_dim)
            for col in self._kept_cat_names
        })
        input_dim = cat_embed_dim * len(self._kept_cat_names) + schema.n_continuous

        # Input projection to backbone hidden_size (matches MambaForecaster
        # pattern; allows sLSTMCell layer 0 to use hidden_size → hidden_size
        # weights for uniform param budget).
        self.in_proj = nn.Linear(input_dim, hidden_size)

        # Stack of sLSTM cells.
        self.cells = nn.ModuleList([
            sLSTMCell(
                input_size=hidden_size,
                hidden_size=hidden_size,
                forget_gate=forget_gate,
            )
            for _ in range(n_layers)
        ])

        # Classifier head: same layers + names as ForecasterV2.
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, fc_hidden)
        self.relu = nn.ReLU(inplace=False)
        self.head = nn.Linear(fc_hidden, 1)

        if task == "regression":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
    ) -> torch.Tensor:
        """x_cat: (B, L, n_cat) int64. x_cont: (B, L, n_cont) float32. → (B, 1)."""
        # Build per-timestep input via categorical embeddings + continuous concat.
        cats = []
        for i, col in enumerate(self.schema.categorical):
            if col in self.drop_categorical:
                continue
            cats.append(self.embeddings[col](x_cat[..., i]))   # (B, L, embed_dim)
        x = torch.cat(cats + [x_cont], dim=-1) if cats else x_cont
        x = self.in_proj(x)                                    # (B, L, H)

        # Roll through time with each sLSTMCell layer.
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        states = [cell.init_state(B, device, dtype) for cell in self.cells]

        h_t = None
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_t, states[layer_idx] = cell(x_t, states[layer_idx])
                x_t = h_t   # output of this layer → input of next layer

        # h_t is the final hidden state from the top layer at the last time step.
        h = self.relu(self.fc(self.dropout(h_t)))
        delta = self.head(h)

        if self.task == "regression" and self._persistence_idx is not None:
            baseline = x_cont[:, -1, self._persistence_idx].unsqueeze(-1)
            return baseline + delta
        return delta
