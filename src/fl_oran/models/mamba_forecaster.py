"""MambaForecaster: ForecasterV2-shaped wrapper around a pure-PyTorch Mamba-S6 backbone.

Mirrors :class:`fl_oran.models.forecaster_v2.ForecasterV2` exactly except that
the temporal trunk is replaced by two stacked :class:`MambaS6Block` layers.
The encoder (categorical embeddings + continuous concat) and the classifier
head (`fc → relu → head`) are kept identical so that any AUC / loss difference
across the two models is attributable to the backbone.

Why a pure-PyTorch implementation? Per ADR-001 D-20 dep-sanity outcome
(2026-04-25), the upstream `mamba-ssm` package cannot be built in our
environment (PyTorch 2.10 + CUDA 12.8 runtime, no system `nvcc`, no pre-built
cu128 wheel on PyPI). The selective-scan algorithm from Gu & Dao 2024
("Mamba: Linear-Time Sequence Modeling with Selective State Spaces") §3.5
is short enough to re-implement in-tree; the sequential scan is ~2-3× slower
than the upstream Triton kernel but functionally identical and removes a
fragile system dependency from the reproducibility artifact.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..data_v2.encoders import FeatureSchema


class MambaS6Block(nn.Module):
    """Pure-PyTorch implementation of the Mamba-S6 selective state-space block.

    Implements algorithm 2 of Gu & Dao 2024 with a sequential scan
    (no parallel-scan kernel). For ``seq_len=5`` the constant-factor
    overhead of the Python loop is negligible; for longer sequences a
    parallel-scan implementation would be preferable.

    Args:
        d_model: input/output channel dimension.
        d_state: SSM hidden-state dimension N.
        d_conv: 1-D causal-conv kernel width.
        expand: inner expansion factor (mirrors mamba-ssm convention).
        dt_rank: rank of the dt low-rank projection. ``"auto"`` uses
            ``ceil(d_model / 16)`` per the upstream default.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            assert isinstance(dt_rank, int)
            self.dt_rank = dt_rank

        # Up-projection to (x, z) gates.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # Depthwise causal 1-D conv on x branch.
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        # Project x to (dt_low_rank, B, C).
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        # dt_proj lifts the low-rank dt back to d_inner.
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # State matrix A is real, log-parameterised, negative.
        # Initialise A_log so that A = -[1, 2, ..., d_state] per channel
        # (the S4D-Real initialisation used by mamba-ssm).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        # D is the skip path (one scalar per channel).
        self.D = nn.Parameter(torch.ones(self.d_inner))
        # Final down-projection back to d_model.
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)."""
        b, length, _ = x.shape
        xz = self.in_proj(x)            # (B, L, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Depthwise causal conv (trim the right padding).
        x_conv = x_branch.transpose(1, 2)                   # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :length]          # causal trim
        x_conv = F.silu(x_conv).transpose(1, 2)             # (B, L, d_inner)

        # Selective parameters: dt, B, C are input-dependent.
        x_dbl = self.x_proj(x_conv)                         # (B, L, dt_rank + 2*d_state)
        dt_low, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_low))               # (B, L, d_inner)

        A = -torch.exp(self.A_log)                          # (d_inner, d_state)
        y = self._selective_scan(x_conv, dt, A, B_param, C_param, self.D)

        # Gate by SiLU(z), then down-project.
        return self.out_proj(y * F.silu(z_branch))

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        # x:  (B, L, d_inner)        dt: (B, L, d_inner)
        # A:  (d_inner, d_state)     B/C: (B, L, d_state)
        # D:  (d_inner,)
        b, length, d_inner = x.shape
        d_state = A.shape[1]

        # Discretise (zero-order hold approximation):
        # dA[b,l,e,n] = exp(dt[b,l,e] * A[e,n])
        # dBx[b,l,e,n] = dt[b,l,e] * B[b,l,n] * x[b,l,e]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dBx = (dt.unsqueeze(-1) * B_param.unsqueeze(2)) * x.unsqueeze(-1)

        h = x.new_zeros(b, d_inner, d_state)
        outputs = []
        for t in range(length):
            h = dA[:, t] * h + dBx[:, t]                        # (B, d_inner, d_state)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)                          # (B, L, d_inner)

        # Skip path: D acts per-channel on the conv output.
        return y + D.unsqueeze(0).unsqueeze(0) * x


class MambaForecaster(nn.Module):
    """Drop-in alternative to ForecasterV2 with a Mamba-S6 backbone."""

    def __init__(
        self,
        schema: FeatureSchema,
        task: Literal["regression", "classification"],
        seq_len: int = 5,
        *,
        cat_embed_dim: int = 8,
        backbone_d_model: int = 64,
        backbone_d_state: int = 16,
        backbone_d_conv: int = 4,
        backbone_expand: int = 1,
        n_blocks: int = 2,
        fc_hidden: int = 64,
        dropout: float = 0.1,
        out_proj_dim: int = 32,
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

        # Encoder: same convention as ForecasterV2 (n+1 rows per cat for OOV).
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(schema.categorical_sizes[col] + 1, cat_embed_dim)
            for col in schema.categorical
        })
        input_dim = cat_embed_dim * schema.n_categorical + schema.n_continuous

        # Trunk: input projection → N Mamba blocks at d_model → output projection.
        self.in_proj = nn.Linear(input_dim, backbone_d_model)
        self.blocks = nn.ModuleList([
            MambaS6Block(
                d_model=backbone_d_model,
                d_state=backbone_d_state,
                d_conv=backbone_d_conv,
                expand=backbone_expand,
            )
            for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(backbone_d_model, out_proj_dim)

        # Classifier head: identical layers + names as ForecasterV2.
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_proj_dim, fc_hidden)
        self.relu = nn.ReLU(inplace=False)
        self.head = nn.Linear(fc_hidden, 1)

        if task == "regression":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """x_cat: (B, L, n_cat) int64. x_cont: (B, L, n_cont) float32. → (B, 1)."""
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
