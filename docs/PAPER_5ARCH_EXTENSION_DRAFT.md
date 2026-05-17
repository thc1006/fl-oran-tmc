# Paper 5-arch extension draft (xLSTM + Mamba-3)

**Status**: paste-ready paper text once V100 extended sweep (#40) completes
and Path D + extended aggregator (`scripts/aggregate_path_d.py`) has emitted
`docs/RESULTS_V7_PATH_D_EXTENDED.md`. The 3-arch core paper (v0.9.2-submission-
ready, Zenodo DOI 10.5281/zenodo.20075433) stands as-is; this extension is
**additive**: the 3-arch narrative arc remains the spine, with the 5-arch
panel as a §7.x reviewer-anticipation section that addresses the "recent
SOTA" axis without re-litigating the core findings.

## Why a separate-section approach (vs full rewrite)

The 3-arch sweep (Phase 5, 900 cells: LSTM × Mamba × Spiking-SSM, fully
analysed) is the paper's load-bearing evidence for the four core claims
(natural-by-BS dominance, algorithmic-gain ceiling, SCAFFOLD × Mamba
interaction, architecture-energy leverage). Rewriting §1, §4, §6, §7 to be
5-arch would (a) inflate the diff substantially before JSAC submission, (b)
risk paired-bootstrap CI95 mismatches (xLSTM/Mamba-3 lack Phase 5
FedAvg/FedAdam baselines — see `experiments/specs/path_d_full.yaml` Path D
extension caveat), and (c) obscure the core narrative with a "kitchen-sink"
arch panel. Instead this extension lives as:

- 1 line in §3 (Dataset) acknowledging 5 backbones evaluated.
- 2 new paragraphs in §4.1 (Method) describing xLSTM-sLSTM and Mamba-3
  with the same encoder/head shell.
- 1 new paragraph in §4.1 explaining param-count parity for the 2 new
  archs (xlstm = 43241, mamba3 = 40635 vs LSTM 44553; within ±10%).
- 1 new §7.x section reporting the 5-arch extended-sweep results with
  paired-bootstrap CI95 deltas where Phase 5 baselines exist (i.e.,
  on FedAvg vs FedSCAM-class algos for the 2 new archs, unpaired
  against the original 3 archs).
- 1 new §future-work bullet on the 5-arch direction.

## §3 Dataset — 1-line extension (paste at end of `\subsection{Preprocessing pipeline}`)

```latex
We benchmark a 3-architecture core panel (LSTM, Mamba, Spiking-SSM;
Section~\ref{sec:method}) on this corpus and additionally report a 2-arch
recent-SOTA extension (xLSTM, Mamba-3; Section~\ref{sec:5arch-extension})
following the same preprocessing pipeline.
```

## §4.1 Method — replacement subsection (drop in for current "Three architectures")

The current §4.1 is titled "Three architectures with a shared encoder--classifier shell". Keep that section verbatim for the core paper, and **add the following new subsection immediately after** the Spiking-SSM paragraph:

```latex
\subsection{Two recent-SOTA additions: xLSTM-sLSTM and Mamba-3}\label{sec:5arch-additions}

To address the reviewer dimension ``does the paper's claim generalise to
the most recent state-of-the-art sequence backbones?'', we add two further
architectures to the panel, both wrapping the same encoder--classifier
shell as the core 3 archs so any AUC delta is attributable to the temporal
trunk only. The core 3-arch findings (Sections~\ref{sec:results} and
\ref{sec:discussion}) are reported on Phase 5 (900 cells, 3 archs $\times$
5 algorithms $\times$ 6 partitions $\times$ 10 seeds, FedAvg + FedAdam
baselines for paired-bootstrap CI$_{95}$); the 5-arch panel reports on the
Path D extended sweep (360 additional cells, 2 new archs $\times$ 3
SAM-family algorithms $\times$ 6 partitions $\times$ 10 seeds) and is
analysed in Section~\ref{sec:5arch-extension}.

\textbf{xLSTM-sLSTM} (\lstinline|xLSTMForecaster|, Beck et
al.~\cite{Beck2024_xLSTM}). Single-head sLSTM (scalar memory) with three
extensions over the canonical 1997 LSTM cell: (a)~exponential input gate
$i_t = \exp(\tilde{\imath}_t)$ allowing tokens to up-weight memory writes
beyond the sigmoid ceiling; (b)~normalizer state $n_t = f_t n_{t-1} + i_t$
that compensates for the unbounded exponential input by dividing the
hidden output ($h_t = o_t \cdot c_t / \max(\lvert n_t \rvert,
\exp(-m_t))$); (c)~stabilizer state $m_t = \max(\log f_t + m_{t-1}, \log
i_t)$ that holds the per-step running maximum of pre-activations so the
stabilised gates $i'_t = \exp(\tilde{\imath}_t - m_t)$ and $f'_t =
\exp(\log f_t + m_{t-1} - m_t)$ stay bounded below $1$ without affecting
the recurrence's mathematical output (paper eq.~15--17). We use the sLSTM
(scalar) memory variant rather than mLSTM (matrix) because at our
\lstinline|seq_len=5| the associative-recall advantage of mLSTM does not
materialise and its extra parameter cost would breach the $\pm 10\%$
parity budget. Two stacked sLSTM cells at \lstinline|hidden_size=48|; the
input projection \lstinline|nn.Linear(input -> 48)| matches the
ForecasterV2 / MambaForecaster pattern. Backbone exposes $b=48$.

\textbf{Mamba-3} (\lstinline|Mamba3Forecaster|, Lahoti et
al.~\cite{Lahoti2026_Mamba3}). Selective state-space block extending
Mamba-2~\cite{GuDao2023_Mamba} with two of the paper's three innovations
(MIMO state updates are deferred as an LLM-scale hardware-utilisation
optimisation not needed at our 40K-parameter scale): (a)~\emph{exponential-
trapezoidal discretisation} (paper Proposition~1), which augments the
Mamba-2 recurrence $h_t = \alpha_t h_{t-1} + \gamma_t B_t x_t$ with a
previous-input contribution $\beta_t B_{t-1} x_{t-1}$ where $\alpha_t =
\exp(\Delta_t A_t)$, $\beta_t = (1 - \lambda_t) \Delta_t \exp(\Delta_t
A_t)$, $\gamma_t = \lambda_t \Delta_t$, and $\lambda_t \in [0, 1]$ is a
data-dependent trapezoidal mixing parameter (a fresh
\lstinline|Linear(d_inner -> 1) + sigmoid| head per block);
(b)~\emph{complex-valued SSM via RoPE-style $2{\times}2$ rotation} (paper
Proposition~2), which pairs adjacent real state dimensions $(h_{2k},
h_{2k+1})$ as one complex state $h_k^{\mathbb{C}} = h_{2k} + i \cdot
h_{2k+1}$ and applies rotation $R(\theta_t) \cdot \rho_t$ each step, where
$\rho_t = \exp(\Delta_t A_t)$ is the real decay (per-channel-per-pair) and
$\theta_t = \mathit{theta\_proj}(x_t)$ is a data-dependent rotation angle
(per complex pair, shared across the $d_\mathrm{inner}$ channels). Two
stacked Mamba-3 blocks at $(d_\mathrm{model}=64, d_\mathrm{state}=16$
$\Rightarrow 8$ complex pairs$, \mathit{expand}=1)$. We initialise
$\mathit{lambda\_proj.bias} = +3.0$ so $\sigma(3) \approx 0.95$ keeps the
block near Mamba-2 Euler at initialisation (the paper's Remark~3
recommends not enforcing the textbook $\lambda_t = 1/2 + O(\Delta t)$
constraint), and $\mathit{theta\_proj} = \mathbf{0}$ at init so the
rotation is the identity until the model learns otherwise. Backbone
exposes $b=64$.

\paragraph{Parameter-count parity (\textsection{}3-arch core: pinned;
\textsection{}5-arch extension: same constraint).}
The $\pm 10\%$ parameter-count parity constraint of the core panel (LSTM
44\,553 / Mamba 40\,489 / Spiking-expand2 43\,593, see
\lstinline|tests/test_v7_fl_arch_agnostic.py| pin tests) extends to the 5-arch
panel: xLSTM 43\,241 (\(-3.0\%\) vs LSTM) and Mamba-3 40\,635 (\(-8.8\%\)
vs LSTM). Pin tests for both new archs guarantee schema-drift fail-loud
behaviour identical to the original 3. Architecture-level confounder
note: ForecasterV2 uses a bottleneck \lstinline|LSTM(input -> 64) ->
LSTM(64 -> 32)| structure while xLSTMForecaster uses a uniform-width
sLSTM(48 $\to$ 48) $\to$ sLSTM(48 $\to$ 48) structure; total
parameter count is within $\pm 10\%$ but layer-wise capacity distribution
differs. We control architecture-level confounders only via total-
parameter-count parity per the preregistered design constraint.

\paragraph{Implementation pragmatics.}
xLSTM's stabilizer is implemented per the Beck et al.\ paper's
mathematical-equivalence guarantee: the unstabilised hidden update is
$h_t = o_t \cdot c_t / \max(\lvert n_t \rvert, 1)$, which under the
stabiliser scaling becomes $h_t = o_t \cdot c'_t / \max(\lvert n'_t
\rvert, \exp(-m_t))$ (NOT $\max(\lvert n'_t \rvert, 1)$ --- this was a
subtle stabilization bug we caught during pre-PR review; the corrected
form preserves the paper-eq.~10 unstabilised limit and is pin-tested by
\lstinline|test_n_safe_uses_exp_minus_m_not_one|). Mamba-3's complex
state is represented as paired real channels rather than
\lstinline|torch.complex64| tensors, following the paper's
Proposition~2 RoPE equivalence; this preserves \lstinline|bf16|
compatibility (\lstinline|torch.complex| has limited \lstinline|bf16|
support across CUDA versions). Both backbones use a pure-PyTorch
sequential scan, like our Mamba-S6 implementation, so the reproducibility
artefact requires only a PyTorch + CUDA runtime, not custom Triton
kernels.
```

## §7.x Results — new section (after the 3-arch results discussion)

```latex
\section{Recent-SOTA extension: 5-arch panel}\label{sec:5arch-extension}

We complement the 3-arch core results (Section~\ref{sec:results}) with a
5-arch extension reporting xLSTM-sLSTM and Mamba-3 on the same Path D
SAM-family algorithm suite (FedSCAM~\cite{FedSCAM2026},
FedGMT~\cite{FedGMT2025}, FedMoSWA~\cite{FedMoSWA2025}) used by the core
SAM-family analysis in Section~\ref{sec:sam-family-empirical}. The
extension sweep adds 360 cells = 2 archs $\times$ 3 algorithms $\times$ 6
partitions $\times$ 10 seeds, executed on $4 \times$ V100-SXM2-32GB as a
post-pilot decision (per CHECKPOINT 3 of
\lstinline|experiments/specs/path_d_extended_pilot.yaml| --- the 4-cell
pilot at IID confirmed GO criteria for both new archs).

\paragraph{Scope of paired statistics.}
Phase 5 ran the original 3 archs only, so FedAvg / FedAdam baselines for
xLSTM and Mamba-3 are not available in our current artefact set. We
therefore report \textbf{absolute test AUC} for the 2 new archs across
all 18 (algorithm, partition) cells with 10-seed mean $\pm$ std, and
\textbf{paired-bootstrap CI$_{95}$} only for within-arch algorithm
comparisons (e.g., xLSTM $\times$ FedSCAM vs xLSTM $\times$ FedGMT at the
same partition / seed). A future Phase 5 extension covering xLSTM and
Mamba-3 with the original 5 algorithms (FedAvg / FedProx / FedAdam /
SCAFFOLD / FedDyn) would enable the analogous paired comparison; we
flag this as deferred follow-up rather than a finished result.

\paragraph{High-level findings.}
[PLACEHOLDER --- to be populated after `scripts/aggregate_path_d.py`
emits the 5-arch table. Anticipated structure:
  - xLSTM relative to LSTM at each (algo, partition): unpaired
    delta + Welch CI$_{95}$.
  - Mamba-3 relative to Mamba at each (algo, partition): unpaired
    delta + Welch CI$_{95}$.
  - Energy delta (NVML kJ per cell) for 2 new archs on V100 (note:
    V100 is sm\_70, our Phase 5 used RTX~4080 sm\_89, so energy
    numbers are not directly comparable across hardware; we report
    V100-side only and flag this as a hardware-confounder caveat).
]

\paragraph{What this extension does and does not claim.}
The 5-arch extension is empirically additive: it tests whether
xLSTM-sLSTM and Mamba-3 reproduce the qualitative core-paper findings
(architecture-energy leverage dominating algorithm leverage; natural-by-BS
partition outperforming Dirichlet) on the same dataset. It does
\emph{not} re-litigate the 4 core claims of Section~\ref{sec:results} ---
those rest on the 900-cell 3-arch Phase 5 plus the 540-cell Path D
SAM-family extension already analysed in Section~\ref{sec:sam-family-empirical}.
The 5-arch panel's primary purpose is to address the JSAC reviewer
dimension of recency.
```

## §future-work — 1 new bullet (insert into existing future-work list)

```latex
  \item \textbf{Per-architecture per-partition extension on full Phase 5
    algorithm suite}: the 5-arch extension (Section~\ref{sec:5arch-extension})
    reports xLSTM and Mamba-3 on the Path D SAM-family algorithms only.
    Extending Phase 5 (FedAvg / FedProx / FedAdam / SCAFFOLD / FedDyn)
    to the 2 new archs would enable a full $5 \times 5$ paired-bootstrap
    CI$_{95}$ algorithm comparison across all backbones, including the
    architecture $\times$ algorithm catastrophic-interaction cell
    (currently characterised on Mamba $\times$ SCAFFOLD only --- whether
    Mamba-3 inherits or escapes that interaction is an open question
    motivated by its different state representation, see Lahoti et
    al.~\cite{Lahoti2026_Mamba3}).
```

## §threats-to-validity (optional 1-paragraph addition)

```latex
\paragraph{Cross-hardware energy comparison in the 5-arch extension.}
The 3-arch core panel's energy results (Section~\ref{sec:arch-leverage})
are measured on RTX~4080 (sm\_89, native \lstinline|bf16|). The 5-arch
extension's V100 cells use sm\_70 with emulated \lstinline|bf16|, which
is approximately $30\%$ slower than fp16 but stable without
\lstinline|GradScaler|. Energy comparisons \emph{across} the
core-vs-extension boundary are therefore not directly meaningful; we
report V100-side energy numbers for the 2 new archs separately and do
not attempt to combine them with the RTX~4080 numbers. A
hardware-uniform extended sweep on a single GPU class would be
necessary for cross-arch energy ranking across all 5 backbones.
```

## Placement summary

| Section | Action | Lines added |
|---|---|---|
| §3 (Dataset) | 1-line trailer in `\subsection{Preprocessing pipeline}` | 3 |
| §4.1 (Method) | New subsection after Spiking-SSM paragraph | ~70 |
| §4.1 | Param-count paragraph extension | ~14 |
| §4.1 | Implementation pragmatics paragraph | ~16 |
| §7 (Results) | New `\section{Recent-SOTA extension}` | ~35 (+ placeholder for data) |
| §future-work | 1 new bullet | ~10 |
| §threats | 1 paragraph on hardware | ~9 |
| **Total** | | **~157 lines paper text** |

## Pre-merge checklist (when ready to paste into main.tex)

- [ ] V100 extended sweep (#40) complete; cells in `artifacts/v7_sam_family/`
  for the 2 new archs
- [ ] `scripts/aggregate_path_d.py` regenerated against 5-arch cells (the
  PATH_D_ARCHS fix from PR #20 already covers this)
- [ ] §7.x "High-level findings" PLACEHOLDER replaced with concrete numbers
- [ ] Per-arch wall-time numbers in `scripts/sweep_dashboard.py`
  ARCH_WALL_FALLBACK_S refined post-pilot (PR #22)
- [ ] PDF rebuild (`cd paper && pdflatex main && bibtex main && pdflatex
  main && pdflatex main`) — check that all 3 new bib citations resolve
  (`Beck2024_xLSTM`, `Lahoti2026_Mamba3`, `Alharthi2024_xLSTMTime`)
- [ ] Visual review of param-count table parity claims (all 5 numbers
  match the pin tests)
