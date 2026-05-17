"""Mamba3Forecaster: ForecasterV2-shaped wrapper around a pure-PyTorch Mamba-3 backbone.

Source: Lahoti et al., "Mamba-3: Improved Sequence Modeling using State Space
Principles", arXiv:2603.15569, 16 Mar 2026. We implement two of the three
innovations the paper introduces (the third — MIMO state updates — is a
hardware-utilization optimization that doesn't apply at our 40K-param scale,
deferred per ``docs/PAPER_NOTES_MAMBA3.md``):

**Innovation 1 — Exponential-Trapezoidal Discretization (§3.1, Proposition 1).**
Mamba-2 uses ``h_t = α_t · h_{t-1} + γ_t · B_t · x_t`` with ``α_t =
exp(Δ_t · A_t)`` and ``γ_t = Δ_t`` (exponential-Euler). Mamba-3 adds a
previous-input contribution::

    h_t = α_t · h_{t-1}  +  β_t · B_{t-1} · x_{t-1}  +  γ_t · B_t · x_t
    α_t = exp(Δ_t · A_t)
    β_t = (1 - λ_t) · Δ_t · exp(Δ_t · A_t)
    γ_t = λ_t · Δ_t
    λ_t ∈ [0, 1] data-dependent (trapezoidal mixing parameter)

Setting ``λ_t = 1`` recovers Mamba-2. Setting ``λ_t = 1/2`` recovers the
classical trapezoidal rule. The paper's Remark 3 reports that NOT enforcing
the textbook ``λ_t = 1/2 + O(Δt)`` constraint gives better empirical
performance, so ``λ_t`` is a free data-dependent scalar produced by a
``Linear(d_inner, 1) → sigmoid`` head.

**Innovation 2 — Complex-Valued SSM via RoPE-style 2x2 Rotation (§3.2).**
Mamba-2 uses real eigenvalues, restricting state dynamics to monotonic decay.
Proposition 2 shows a complex SSM can be expressed as a real SSM whose state
transitions are *block-diagonal 2x2 rotation matrices* (Su et al. 2023's RoPE
trick), avoiding the need for ``torch.complex64`` tensors. Each pair of
adjacent real state dims (``h[..., 2k]``, ``h[..., 2k+1]``) is treated as one
complex state ``h_k = h[..., 2k] + i·h[..., 2k+1]`` and updated as::

    [h_re_new]   [cos θ, -sin θ]   [h_re_prev]
    [h_im_new] = [sin θ,  cos θ] * [h_im_prev]   * ρ

with ``ρ_t = exp(Δ_t · A)`` (real decay, per-channel-per-pair, identical to
Mamba-2's α magnitude) and ``θ_t = theta_proj(x_t)`` (data-dependent
rotation angle, shared across d_inner channels, per complex pair).

The data-dependence of ``θ_t`` is what differentiates Mamba-3 from a
learned-static complex A: rotation can switch at runtime based on the input,
giving the SSM rotational-state expressivity that real-eigenvalue Mamba-2
cannot represent (e.g., parity counting, alternating patterns).

**Design choices (audited 2026-05-18 deep-review):**

* ``d_state`` must be EVEN (complex pairs); we default to 16 = 8 pairs.
* ``A_log`` shrinks from Mamba-2's ``(d_inner, d_state)`` to
  ``(d_inner, d_state // 2)`` because each complex pair shares one ρ.
* ``theta_proj.bias = 0`` at init so that θ ≈ 0 ≈ identity rotation;
  Mamba-3 starts close to a real-eigenvalue baseline and learns rotation
  during training.
* ``lambda_proj.bias = +3.0`` at init so that ``sigmoid(3) ≈ 0.95 ≈ 1``,
  meaning Mamba-3 starts close to Mamba-2's Euler rule and learns the
  trapezoidal contribution during training. This is a conservative
  initialization — adversarial inits like ``lambda_proj.bias = 0`` would
  start at the trapezoidal midpoint, which is not what Remark 3 recommends.

Param count budget (vs Mamba-2 at d_model=64, expand=1, d_state=16):
``+ lambda_proj(65) + theta_proj(520) − A_log reduction (−512) ≈ +73 per
block``. With 2 blocks, total backbone ~33K + encoder/head ~10K ≈ **~43K**,
within ±10% of ForecasterV2 per ADR-001 D-20.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..data_v2.encoders import FeatureSchema


class Mamba3SSMBlock(nn.Module):
    """Pure-PyTorch Mamba-3 block (SISO, Innovations 1 + 2 from arXiv:2603.15569).

    Replaces :class:`fl_oran.models.mamba_forecaster.MambaS6Block`'s
    selective scan with the trapezoidal + complex-rotation recurrence
    described in the module docstring. Forward signature is identical so
    Mamba-3 can be a drop-in trunk component.

    Args:
        d_model: input/output channel dimension.
        d_state: SSM hidden-state dimension N. MUST be even (complex
            pairing); a ValueError is raised otherwise.
        d_conv: 1-D causal-conv kernel width.
        expand: inner expansion factor (mirrors mamba-ssm convention).
        dt_rank: rank of the dt low-rank projection. ``"auto"`` uses
            ``ceil(d_model / 16)`` per the upstream Mamba-2 default.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        dt_rank: int | str = "auto",
    ) -> None:
        super().__init__()
        if d_state % 2 != 0:
            raise ValueError(
                f"Mamba3SSMBlock requires even d_state for complex pairing; "
                f"got d_state={d_state}"
            )
        self.d_model = d_model
        self.d_state = d_state
        self.d_state_pairs = d_state // 2
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            assert isinstance(dt_rank, int)
            self.dt_rank = dt_rank

        # --- Mamba-2 lineage (in_proj / conv / x_proj / dt_proj / D / out_proj). ---
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A_log: one decay magnitude per (channel, complex pair). HALF the
        # size of Mamba-2's A_log because each pair shares one ρ across its
        # two real dims (re/im). Init = log(arange(1, d_state_pairs+1)) so
        # that A = -exp(A_log) gives the S4D-Real decay schedule.
        A_init = torch.arange(1, self.d_state_pairs + 1, dtype=torch.float32)
        A_init = A_init.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A_init))

        # D: skip path (one scalar per channel) — unchanged from Mamba-2.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # --- Mamba-3 NEW: lambda_proj + theta_proj. ---
        # Innovation 1: data-dependent trapezoidal mix λ_t ∈ [0, 1].
        # bias = +3 makes sigmoid(3) ≈ 0.953, so the block starts near
        # λ ≈ 1 (Mamba-2 Euler) and learns trapezoidal during training.
        self.lambda_proj = nn.Linear(self.d_inner, 1, bias=True)
        nn.init.zeros_(self.lambda_proj.weight)
        nn.init.constant_(self.lambda_proj.bias, 3.0)

        # Innovation 2: data-dependent rotation angle θ_t per complex pair.
        # bias = 0 + weight = 0 gives θ ≡ 0 ≈ identity rotation at init.
        # The model learns to use rotation during training; we don't bake
        # in any particular angle a priori.
        self.theta_proj = nn.Linear(
            self.d_inner, self.d_state_pairs, bias=True,
        )
        nn.init.zeros_(self.theta_proj.weight)
        nn.init.zeros_(self.theta_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)."""
        b, length, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Depthwise causal conv (same as Mamba-2).
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[..., :length]
        x_conv = F.silu(x_conv).transpose(1, 2)            # (B, L, d_inner)

        # Selective parameters: dt, B, C (Mamba-2 lineage).
        x_dbl = self.x_proj(x_conv)
        dt_low, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_low))               # (B, L, d_inner)

        # Mamba-3 NEW: data-dependent λ_t and θ_t (whole sequence at once).
        lambda_seq = torch.sigmoid(self.lambda_proj(x_conv))  # (B, L, 1)
        theta_seq = self.theta_proj(x_conv)                   # (B, L, d_state_pairs)

        A = -torch.exp(self.A_log)                          # (d_inner, d_state_pairs)
        y = self._selective_scan(
            x_conv, dt, A, B_param, C_param, self.D, lambda_seq, theta_seq,
        )

        return self.out_proj(y * F.silu(z_branch))

    @staticmethod
    def _rotate_and_decay(
        z: torch.Tensor,
        rho: torch.Tensor,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
    ) -> torch.Tensor:
        """Apply per-pair 2x2 rotation + decay to a real-valued state tensor.

        Args:
            z: state, shape ``(B, d_inner, d_state)``. Adjacent pairs
                ``(z[..., 2k], z[..., 2k+1])`` are treated as complex state
                ``z_k^C = z_re + i·z_im``.
            rho: per-(channel, pair) magnitude in ``(0, 1)``, shape
                ``(B, d_inner, d_state_pairs)``.
            cos_t, sin_t: per-pair rotation cos/sin, shape
                ``(B, d_state_pairs)`` (shared across d_inner channels per
                paper §3.2).

        Returns:
            Rotated-and-decayed state, shape ``(B, d_inner, d_state)``.
        """
        b, d_inner, d_state = z.shape
        z_pairs = z.view(b, d_inner, d_state // 2, 2)
        z_re = z_pairs[..., 0]                              # (B, d_inner, pairs)
        z_im = z_pairs[..., 1]

        # Broadcast cos/sin across the d_inner channel axis.
        cos_b = cos_t.unsqueeze(1)                          # (B, 1, pairs)
        sin_b = sin_t.unsqueeze(1)

        new_re = rho * (cos_b * z_re - sin_b * z_im)
        new_im = rho * (sin_b * z_re + cos_b * z_im)
        return torch.stack([new_re, new_im], dim=-1).view(b, d_inner, d_state)

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
        D: torch.Tensor,
        lambda_seq: torch.Tensor,
        theta_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Mamba-3 trapezoidal selective scan with complex-state rotation.

        Recurrence at each step t (paper §3.1, eq 5-6):

            ρ_t   = exp(Δ_t · A)                              # decay magnitude
            θ_t   = theta_seq[:, t]                           # rotation angle (data-dep)
            rot   = block-diag(R(θ_t))                        # 2x2 rotation per pair
            α_t   = ρ_t * rot                                 # complex transition
            γ_t·B·x_t  = λ_t · Δ_t · B_t · x_t                 # current-step input (no rot)
            β_t·B·x_{t-1} = (1-λ_t) · Δ_t · (α_t · B_{t-1} x_{t-1})
                          = (1-λ_t) · Δ_t · rot · ρ_t · B_{t-1}x_{t-1}

            h_t = α_t · h_{t-1} + β_t·B·x_{t-1} + γ_t·B·x_t

        ``B_x_prev = 0`` at t=0 (sequence-prefix convention).

        Args:
            x: conv output, shape ``(B, L, d_inner)``.
            dt: discretization step Δ_t, shape ``(B, L, d_inner)``.
            A: state matrix (negative, exp-log-parameterised), shape
                ``(d_inner, d_state_pairs)``.
            B_param, C_param: input/output projections, shape
                ``(B, L, d_state)``.
            D: skip-path per-channel scalar, shape ``(d_inner,)``.
            lambda_seq: trapezoidal mix per step, shape ``(B, L, 1)``.
            theta_seq: rotation angle per pair per step, shape
                ``(B, L, d_state_pairs)``.

        Returns:
            Output ``y``, shape ``(B, L, d_inner)``. Includes the
            skip-path ``D · x`` contribution.
        """
        b, length, d_inner = x.shape
        d_state_pairs = A.shape[1]
        d_state = 2 * d_state_pairs

        # ρ_{t,channel,pair} = exp(Δ_t · A[channel, pair]).
        # dt[..., None] broadcasts (B, L, d_inner, 1) against A[None, None]
        # which is (1, 1, d_inner, d_state_pairs) → (B, L, d_inner, d_state_pairs).
        rho_seq = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        cos_seq = torch.cos(theta_seq)                       # (B, L, d_state_pairs)
        sin_seq = torch.sin(theta_seq)

        # Precompute B_t · x_t for each step (no rotation here, just the
        # "delta-input" tensor that gets multiplied by γ or β coefficients).
        # B_param: (B, L, d_state); x: (B, L, d_inner). Want (B, L, d_inner, d_state).
        Bx_seq = B_param.unsqueeze(2) * x.unsqueeze(-1)      # (B, L, d_inner, d_state)

        h = x.new_zeros(b, d_inner, d_state)
        Bx_prev = x.new_zeros(b, d_inner, d_state)
        outputs = []
        for t in range(length):
            rho_t = rho_seq[:, t]                            # (B, d_inner, pairs)
            cos_t = cos_seq[:, t]                            # (B, pairs)
            sin_t = sin_seq[:, t]
            lambda_t = lambda_seq[:, t]                      # (B, 1)
            dt_t = dt[:, t]                                  # (B, d_inner)
            Bx_t = Bx_seq[:, t]                              # (B, d_inner, d_state)

            # α_t · h_{t-1} (rotation + decay applied to previous state).
            rotated_h = self._rotate_and_decay(h, rho_t, cos_t, sin_t)

            # γ_t · B_t · x_t = λ_t · Δ_t · B_t · x_t (current-step input,
            # NO rotation per paper §3.1 derivation — the γ coefficient is
            # purely real).
            gamma_scale = (lambda_t * dt_t).unsqueeze(-1)    # (B, d_inner, 1)
            gamma_term = gamma_scale * Bx_t

            # β_t · B_{t-1} · x_{t-1} = (1-λ_t) · Δ_t · α_t · B_{t-1}x_{t-1}
            # α_t = rotation + decay, applied to PREVIOUS-step Bx.
            # At t=0, Bx_prev is zero (sequence-prefix convention) so this
            # term is zero — no branching needed.
            rotated_prev_Bx = self._rotate_and_decay(
                Bx_prev, rho_t, cos_t, sin_t,
            )
            beta_scale = ((1.0 - lambda_t) * dt_t).unsqueeze(-1)
            beta_term = beta_scale * rotated_prev_Bx

            h = rotated_h + gamma_term + beta_term

            # Output: y_t = C_t · h_t (real-valued inner product).
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)   # (B, d_inner)
            outputs.append(y_t)

            Bx_prev = Bx_t

        y = torch.stack(outputs, dim=1)                       # (B, L, d_inner)
        # Skip path: D acts per-channel on the conv output (same as Mamba-2).
        return y + D.unsqueeze(0).unsqueeze(0) * x


class Mamba3Forecaster(nn.Module):
    """Drop-in alternative to ForecasterV2 with a Mamba-3 backbone.

    Encoder (categorical embeddings + continuous concat) and head
    (``fc → relu → head``) are identical to ForecasterV2 / MambaForecaster
    so that AUC differences are attributable to the temporal trunk only.
    """

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

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(schema.categorical_sizes[col] + 1, cat_embed_dim)
            for col in schema.categorical
        })
        input_dim = cat_embed_dim * schema.n_categorical + schema.n_continuous

        self.in_proj = nn.Linear(input_dim, backbone_d_model)
        self.blocks = nn.ModuleList([
            Mamba3SSMBlock(
                d_model=backbone_d_model,
                d_state=backbone_d_state,
                d_conv=backbone_d_conv,
                expand=backbone_expand,
            )
            for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(backbone_d_model, out_proj_dim)

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
