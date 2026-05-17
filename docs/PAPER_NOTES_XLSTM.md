# xLSTM (sLSTM) Implementation Notes

Source: Beck et al., "xLSTM: Extended Long Short-Term Memory", arXiv:2405.04517v2, 6 Dec 2024.
Reference impl: https://github.com/NX-AI/xlstm

We implement **sLSTM only** (scalar memory variant). mLSTM (matrix memory) is
deferred — at our seq_len=5, the matrix memory's "associative recall" advantage
doesn't materialize.

## sLSTM forward pass (paper eq 8-14)

### Core recurrence

```
c_t = f_t · c_{t-1} + i_t · z_t                    (cell state, eq 8)
n_t = f_t · n_{t-1} + i_t                          (normalizer state, eq 9, NEW)
h_t = o_t · (c_t / n_t)                            (hidden state, eq 10, NEW normalization)

z_t = φ(z̃_t),  z̃_t = w_z^T x_t + r_z h_{t-1} + b_z   (cell input, eq 11)
ĩ_t = w_i^T x_t + r_i h_{t-1} + b_i                   (input gate pre-activation, eq 12)
f̃_t = w_f^T x_t + r_f h_{t-1} + b_f                   (forget gate pre-activation, eq 13)
õ_t = w_o^T x_t + r_o h_{t-1} + b_o                   (output gate pre-activation, eq 14)

i_t = exp(ĩ_t)                                     (input gate, EXPONENTIAL — vs LSTM's sigmoid)
f_t = sigmoid(f̃_t) OR exp(f̃_t)                    (forget gate, choice of either)
o_t = sigmoid(õ_t)                                 (output gate, unchanged from LSTM)
```

φ, ψ = activation functions (typically tanh).

### Stabilizer state (eq 15-17, CRITICAL for numerical stability)

Exponential `i_t = exp(ĩ_t)` can overflow. To avoid this:

```
m_t = max(log(f_t) + m_{t-1}, log(i_t))            (stabilizer state, eq 15)

In log-space:
  log(f_t) = log(sigmoid(f̃_t))  or  f̃_t (if forget uses exp)
  log(i_t) = ĩ_t

Stabilized gates:
  i'_t = exp(ĩ_t - m_t)                            (stab input, eq 16)
  f'_t = exp(log(f_t) + m_{t-1} - m_t)             (stab forget, eq 17)
```

Use `i'_t` and `f'_t` in the cell/normalizer updates (eq 8, 9). Paper proves
this leaves forward output + gradients **mathematically identical** to the
unstabilized version (Appendix A.2).

### Differences vs classic LSTM

| Component | LSTM (1997) | sLSTM (2024) |
|---|---|---|
| Input gate `i_t` | sigmoid | **exp + stabilizer** |
| Forget gate `f_t` | sigmoid | sigmoid OR exp + stabilizer |
| Cell state `c_t` | `f·c + i·z` | same |
| Hidden state `h_t` | `o · ψ(c)` | **`o · (c/n)`** (normalized) |
| Normalizer state `n_t` | — | **NEW: `f·n + i`** |
| Stabilizer state `m_t` | — | **NEW: max(log(f)+m, log(i))** |

## Implementation strategy

### Choice: forget gate variant

Paper supports both sigmoid and exp forget gates. For our setting:
- **sigmoid forget**: more stable, classic LSTM-like dynamics
- **exp forget**: required for parity tasks, more expressive
- **Default to sigmoid** for our initial implementation (lower risk; can ablate
  later if interesting).

### Multi-cell heads (paper §2.2)

> "sLSTM can have multiple heads with memory mixing within each head but not
> across heads."

For our small-scale model:
- **Skip multi-head initially**. Single sLSTM cell, single head.
- Simpler, fewer params, faster to debug.
- Can add multi-head later if accuracy demands.

### Class: `sLSTMCell` and `xLSTMForecaster`

```python
class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Input projections (W_x · x_t)
        self.w_z = nn.Linear(input_size, hidden_size, bias=True)
        self.w_i = nn.Linear(input_size, hidden_size, bias=True)
        self.w_f = nn.Linear(input_size, hidden_size, bias=True)
        self.w_o = nn.Linear(input_size, hidden_size, bias=True)

        # Recurrent projections (R · h_{t-1})
        # NOTE: paper uses block-diagonal R for multi-head, we use full for single head
        self.r_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, state):
        h_prev, c_prev, n_prev, m_prev = state
        # Pre-activations
        z_tilde = self.w_z(x) + self.r_z(h_prev)
        i_tilde = self.w_i(x) + self.r_i(h_prev)
        f_tilde = self.w_f(x) + self.r_f(h_prev)
        o_tilde = self.w_o(x) + self.r_o(h_prev)

        # Sigmoid forget, exp input (with stabilizer)
        log_f = F.logsigmoid(f_tilde)            # log(sigmoid(f̃))
        log_i = i_tilde                          # log(exp(ĩ))
        m = torch.maximum(log_f + m_prev, log_i)
        i_prime = torch.exp(log_i - m)
        f_prime = torch.exp(log_f + m_prev - m)

        # Cell + normalizer updates
        z = torch.tanh(z_tilde)
        c = f_prime * c_prev + i_prime * z
        n = f_prime * n_prev + i_prime

        # Output (with normalization)
        o = torch.sigmoid(o_tilde)
        h = o * (c / torch.clamp(n.abs(), min=1.0))

        return h, (h, c, n, m)


class xLSTMForecaster(nn.Module):
    def __init__(self, schema, task, seq_len=5, *,
                 cat_embed_dim=8, hidden_size=48, n_layers=2,
                 fc_hidden=64, dropout=0.1, out_proj_dim=32):
        # ... encoder same as MambaForecaster ...
        self.input_dim = cat_embed_dim * schema.n_categorical + schema.n_continuous
        self.cells = nn.ModuleList([
            sLSTMCell(
                input_size=self.input_dim if i == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for i in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_size, out_proj_dim)
        # ... head same as MambaForecaster: fc → relu → head ...

    def forward(self, x_cat, x_cont):
        # ... build encoded input (B, L, input_dim) ...
        h_state = [None] * len(self.cells)
        for t in range(self.seq_len):
            x_t = encoded[:, t]
            for layer_idx, cell in enumerate(self.cells):
                if h_state[layer_idx] is None:
                    h_state[layer_idx] = init_state(B, hidden_size, device)
                x_t, h_state[layer_idx] = cell(x_t, h_state[layer_idx])
        # x_t is final hidden at t=L
        h = self.out_proj(x_t)
        # ... fc → relu → head ...
```

### Param count target (~40-50K)

For hidden_size=48, n_layers=2, input_dim≈24 (3 cat × 8 + 0 cont):

Per sLSTMCell:
- `w_z, w_i, w_f, w_o`: 4 × (input_size × 48 + 48) ≈ 4 × (24 × 48 + 48) = 4,800 (layer 0)
                                                  ≈ 4 × (48 × 48 + 48) = 9,408 (layer 1+)
- `r_z, r_i, r_f, r_o`: 4 × 48 × 48 = 9,216 each layer

Total per layer:
- Layer 0: 4,800 + 9,216 = 14,016
- Layer 1: 9,408 + 9,216 = 18,624

2 layers: 14,016 + 18,624 = 32,640
+ encoder embeddings (cat × 8 each, ~few hundred per cat)
+ out_proj, fc, head wrapper: ~5,000

**Total: ~38-42K** ✓ within budget

## Testing strategy

1. **Param count test**: ~38-45K
2. **Forward shape test**: (B, L, d_model) → (B, 1)
3. **Stabilizer test**: pre-activations of 100 (extreme) should NOT produce
   NaN/Inf in output. Verify `i_prime` and `f_prime` stay bounded.
4. **Determinism test**: same input + seed → same output
5. **Gradient flow test**: backward pass yields non-zero grads in stabilizer
   path
6. **Equivalence sanity**: at `m_t = 0`, should behave like un-stabilized
   sLSTM (verify forward output)

## References

- **Paper**: Beck et al., NeurIPS 2024, arXiv:2405.04517
- **Reference impl**: https://github.com/NX-AI/xlstm
- **xLSTMTime**: Alharthi & Mahmood, arXiv:2407.10240 (preprint only, NOT
  published in a peer-reviewed venue) — time-series adaptation, but their
  tweaks are for LTSF (long sequences), may not transfer to our seq_len=5.
  We use vanilla sLSTM.

  **Erratum (2026-05-18)**: an earlier version of this notes file
  mis-attributed this paper to "Tabish, MDPI AI 5(3):1418-1444, 2024".
  Both the author name and the journal venue were wrong. Verified
  against the live arXiv page on 2026-05-18: title is "xLSTMTime:
  Long-term Time Series Forecasting With xLSTM", authors are
  Musleh Alharthi + Ausif Mahmood, preprint only. The bib entry
  (``Alharthi2024_xLSTMTime`` in ``paper/bibliography.bib``) was
  corrected in the same PR as this note revision; if you find any
  other reference to "Tabish, MDPI AI" in the repo, please correct it
  to match.
