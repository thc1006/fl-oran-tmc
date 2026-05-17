# Mamba-3 Implementation Notes

Source: Lahoti et al., "Mamba-3: Improved Sequence Modeling using State Space
Principles", arXiv:2603.15569v1, 16 Mar 2026.

## Three innovations vs Mamba-2

### Innovation 1: Exponential-Trapezoidal Discretization (§3.1)

Mamba-2's recurrence (exponential-Euler discretization):
```
h_t = α_t · h_{t-1} + γ_t · B_t · x_t
α_t = exp(Δ_t · A_t)
γ_t = Δ_t
```

**Mamba-3** (Proposition 1, eq 5-6, §3.1):
```
h_t = α_t · h_{t-1} + β_t · B_{t-1} · x_{t-1} + γ_t · B_t · x_t

where:
  α_t = exp(Δ_t · A_t)                  (same as Mamba-2)
  β_t = (1 - λ_t) · Δ_t · exp(Δ_t · A_t)  (NEW: previous-step contribution)
  γ_t = λ_t · Δ_t                       (current-step contribution)
  λ_t ∈ [0, 1]: data-dependent scalar (trapezoidal mixing parameter)
```

Properties:
- `λ_t = 1/2` → classical trapezoidal rule
- `λ_t = 1` → recovers Mamba-2's Euler rule
- Paper: NOT enforcing `λ_t = 1/2 + O(Δt)` gives better empirical performance (Remark 3)
- Second-order accuracy: error `O(Δ³)` under stability assumptions

**Implementation cost**: ~1 hr
- Add `λ_t` projection (data-dependent scalar): `lambda_proj: Linear(d_inner, 1)`
- Maintain 1-step delayed input: `x_{t-1} = roll(x, shift=1, dim=time)`
- Compute `β_t = (1 - λ_t) * Δ_t * exp(Δ_t * A_t)` and add `β_t · B_{t-1} · x_{t-1}` term

### Innovation 2: Complex-Valued SSM (§3.2)

**Key insight from Proposition 2 (page 7)**: Complex SSMs can be **expressed as
real SSMs with block-diagonal 2×2 rotation matrices**. We do NOT need actual
`torch.complex64` tensors.

Real-valued equivalent:
```
h_t = (block-diag of 2x2 rotation matrices R(θ_t)) · h_{t-1}  +  Δ_t B_t x_t
y_t = C_t^T · h_t  (with rotated B, C via RoPE-style trick)
```

Where each 2×2 block is:
```
R(θ) = [cos(θ), -sin(θ);
        sin(θ),  cos(θ)]
```

θ_t is data-dependent (similar to RoPE rotations in transformers).

The **"RoPE trick" from Su et al. 2023** allows efficient computation with
minimal overhead vs real-valued SSMs.

**Why it works**: complex eigenvalues can represent rotational state dynamics
(e.g., parity counting), which Mamba-2's real-only eigenvalues cannot.

**Implementation cost**: ~2-3 hr
- State dim N → effective N/2 with 2-channel real "complex" parts
- θ_t projection: data-dependent rotation angles
- Apply rotation matrices each step (instead of scalar α_t)
- Adapt B, C projections via RoPE-style sin/cos multiplication

### Innovation 3: MIMO (§3.3, page 8+) — DEFER

Switches from outer-product to matrix-multiplication state update for inference
hardware efficiency. Useful at LLM scale (1.5B params) for FLOPs/latency
trade-off.

**For our 40K param model**: SISO is sufficient. MIMO doesn't change quality,
only hardware utilization. **DEFER to future work**.

## Implementation plan for our setting

### Class: `Mamba3S6Block` (SISO with Innovations 1+2)

```python
class Mamba3S6Block(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=1):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state          # MUST be even (for complex pairing)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)

        # Mamba-2-style projections (existing)
        self.x_proj = nn.Linear(self.d_inner,
                                self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # NEW: λ_t for exponential-trapezoidal
        self.lambda_proj = nn.Linear(self.d_inner, 1, bias=True)

        # NEW: θ_t for complex rotation (data-dependent rotation angles)
        # d_state/2 because each complex dim = 2 real channels
        self.theta_proj = nn.Linear(self.d_inner, d_state // 2, bias=True)

        # A_log: log of decay rates (one per inner-state pair)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: (B, L, d_model)
        ...
        # 1. Project to (x_branch, z_branch)
        # 2. Apply causal conv + SiLU on x_branch
        # 3. Compute Δ, B, C (selective parameters) — same as Mamba-2
        # 4. Compute λ_t (NEW): data-dependent trapezoidal mix
        # 5. Compute θ_t (NEW): data-dependent rotation angles
        # 6. Selective scan with:
        #    - exponential-trapezoidal recurrence (Innovation 1)
        #    - rotation matrix transitions instead of scalar α (Innovation 2)
        # 7. Skip path: D · x
        # 8. Gate by SiLU(z) → out_proj
```

### Param count target (~40-50K)

Approximate budget breakdown (mirror our Mamba's structure with d_model=64,
expand=1, d_state=16):
- `embeddings`: ~3 cat × 8 dim = 24 + categorical_size × 8 params (~few K)
- `in_proj`: 64 × 128 = 8,192
- `conv1d`: depthwise, 64 × 4 + 64 = 320
- `x_proj`: 64 × (dt_rank + 2·16) = ~3,000
- `dt_proj`: dt_rank × 64 = ~700
- **NEW** `lambda_proj`: 64 × 1 = 64
- **NEW** `theta_proj`: 64 × 8 = 512
- `A_log` + `D`: 64 × 16 + 64 = 1,088
- `out_proj`: 64 × 64 = 4,096
- 2 blocks (×2 above): ~36K
- + encoder + head wrapper: ~5K
- **Total: ~40-45K** ✓ within budget

### Risks + mitigations

| Risk | Mitigation |
|---|---|
| Complex rotation breaks autograd | Use cos/sin from PyTorch (well-tested) |
| Numerical issues at small Δ_t | Standard Mamba softplus for Δ |
| λ_t projection saturates | Constrain to (0, 1) via sigmoid |
| θ_t rotation angles too large | Initialize small, let learning find scale |

## Testing strategy

1. **Param count test**: assert `~40K ± 5K` after build_model("mamba3")
2. **Forward shape test**: `forward((B, L, d_model)) → (B, L, d_model)`
3. **Gradient flow test**: backward pass yields non-zero grads in lambda_proj
   and theta_proj (NEW params)
4. **Lambda boundary test**: at `λ=1` should approximately recover Mamba-2
   behavior (sanity test)
5. **No-NaN test**: synthetic input with extreme values should not NaN
6. **Determinism test**: same input + seed → same output (cudnn deterministic)
