# Paper §7 SAM-Family Empirical Section (Template)

**Status:** template with `{{PLACEHOLDER}}` markers. Fill in once
V100 60-cell sweep (artifacts/v7_sam_family/) and 4060 KL ablation
(artifacts/v7_fedgmt_kl_off/, artifacts/v7_fedgmt_kl_on/) complete.

**Target placement:** `paper/main.tex` after §7.5 (`sec:why-no-fedswa`,
the existing FedSWA empirical paragraph), as a new subsection
`sec:sam-family-empirical`. The §2.6 related-work paragraph already
defers FedSCAM and FedMoSWA "to follow-up work" — this section
closes that deferral.

---

## §7.6 FedSCAM and FedGMT empirical comparison

\subsection{FedSCAM and FedGMT empirical comparison (deferred $\to$ done)}\label{sec:sam-family-empirical}

The §2.6 related-work mechanism argument predicted that FedSCAM~\cite{FedSCAM2026}
and FedGMT~\cite{FedGMT2025} — both variance-aware sharpness methods that
modify the server-side aggregation step beyond FedAdam's adaptive
$\mathrm{Adam}(m, v)$ damping — would perform within the FedAdam
$+0.006$--$+0.016$ AUC envelope established in §6.3. We close that
empirical gap with a 60-cell sweep on 4$\times$ Tesla V100-SXM2-32GB:
LSTM $\times$ \{FedSCAM, FedGMT\} $\times$ \{IID, Dirichlet $\alpha
\in \{0.05, 0.10, 0.50, 1.0, 5.0\}$\} $\times$ 5 seeds. Both
algorithms use paper-pinned hyperparameters: FedSCAM with
$\rho_{\max} = 0.05$, $\alpha_\rho = 1.0$, $\gamma = 1.0$,
$\beta_{\mathrm{align}} = 0.8$, $\kappa = 1.0$, $B_{\mathrm{pilot}}
= 3$ (paper Algorithm~1 mid-of-tested values); FedGMT with
$\alpha_{\mathrm{EMA}} = 0.95$, $\gamma_{\mathrm{KL}} = 1.0$,
$\tau = 3.0$, $\beta = 10$ (per the reference impl
\verb|harrylee999/FL-SAM| example command).

\paragraph{Headline result (Table~\ref{tab:sam-family-headline}).}
On IID partition, FedSCAM achieves test AUC =
\textbf{\{\{FEDSCAM_IID_AUC_MEAN\}\}} $\pm$ \{\{FEDSCAM_IID_AUC_STD\}\}
(5 seeds), {{FEDSCAM_VS_FEDAVG_NARRATIVE}} the Phase~5 FedAvg
baseline of $0.9159$. FedGMT achieves
\textbf{\{\{FEDGMT_IID_AUC_MEAN\}\}} $\pm$
\{\{FEDGMT_IID_AUC_STD\}\}, {{FEDGMT_VS_FEDADAM_NARRATIVE}} the
FedAdam IID baseline of $0.9178$. {{NAN_COUNT_NARRATIVE: e.g. "All 60
cells completed without NaN" OR "X of 60 cells terminated early via
the \texttt{NonFiniteLossError} guard added in
\texttt{\_local\_loop.py}; remaining cells reported below."}}

Paired-bootstrap CI$_{95}$ on the per-seed AUC delta against Phase~5
baselines (matched IID and Dirichlet partitions):

\begin{table}[t]
\centering
\caption{SAM-family delta vs Phase~5 baselines, LSTM $\times$ \{IID, Dirichlet $\alpha$\} $\times$ 5 seeds. CI$_{95}$ via paired-bootstrap $n_{\mathrm{boot}} = 10000$.}\label{tab:sam-family-vs-baseline}
\begin{tabular}{lccc}
\toprule
Partition & FedSCAM $-$ FedAvg & FedGMT $-$ FedAdam & Wilcoxon $p$ \\
\midrule
IID                            & \{\{D_FEDSCAM_FEDAVG_IID\}\}  & \{\{D_FEDGMT_FEDADAM_IID\}\}  & \{\{P_IID\}\} \\
Dirichlet $\alpha = 0.05$      & \{\{D_FEDSCAM_FEDAVG_A005\}\} & \{\{D_FEDGMT_FEDADAM_A005\}\} & \{\{P_A005\}\} \\
Dirichlet $\alpha = 0.10$      & \{\{D_FEDSCAM_FEDAVG_A010\}\} & \{\{D_FEDGMT_FEDADAM_A010\}\} & \{\{P_A010\}\} \\
Dirichlet $\alpha = 0.50$      & \{\{D_FEDSCAM_FEDAVG_A050\}\} & \{\{D_FEDGMT_FEDADAM_A050\}\} & \{\{P_A050\}\} \\
Dirichlet $\alpha = 1.00$      & \{\{D_FEDSCAM_FEDAVG_A100\}\} & \{\{D_FEDGMT_FEDADAM_A100\}\} & \{\{P_A100\}\} \\
Dirichlet $\alpha = 5.00$      & \{\{D_FEDSCAM_FEDAVG_A500\}\} & \{\{D_FEDGMT_FEDADAM_A500\}\} & \{\{P_A500\}\} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{KL term is \{\{KL_LOAD_BEARING_VERDICT: load-bearing / not load-bearing\}\}.}
To isolate the contribution of FedGMT's KL distillation term against
the Global Model Trajectory (Eq.~(9) of \cite{FedGMT2025}), we ran a
paired KL-ablation on the 4060~Ti: LSTM $\times$ FedGMT $\times$ IID
$\times$ 5 seeds, sample-ratio $0.1$, with $\gamma_{\mathrm{KL}} \in
\{0, 1\}$. The $\gamma_{\mathrm{KL}} = 0$ arm short-circuits the KL
term in the client-side \verb|loss_modifier| closure, reducing the
loss to pure BCE plus FedDyn-style dual
$\frac{1}{\beta}\langle w, h \rangle$. Per-seed AUC deltas
$\Delta_{\mathrm{KL}} = \mathrm{AUC}_{\gamma=1} -
\mathrm{AUC}_{\gamma=0}$:

\begin{itemize}
\item Mean $\Delta_{\mathrm{KL}}$: \{\{DELTA_KL_MEAN\}\}
\item Paired-bootstrap CI$_{95}$: \{\{DELTA_KL_CI95\}\}
\item Same-direction count: \{\{DELTA_KL_SIGN_COUNT\}\} / 5 seeds
\end{itemize}

{{KL_INTERPRETATION_PARAGRAPH: e.g. "The CI$_{95}$ \{\{includes\,/\,excludes\}\} zero, indicating the KL term \{\{has no measurable effect\,/\,is necessary\}\} on this dataset in the IID regime."}}

\paragraph{Mechanism revisited.}
The §2.6 prediction was that FedSCAM and FedGMT would land
\{\{within\,/\,outside\}\} FedAdam's $+0.006$--$+0.016$ envelope.
{{MECHANISM_VERDICT: e.g. "The empirical CI$_{95}$ for the
heterogeneous Dirichlet $\alpha = 0.05$ regime is
[+X.XXXX, +X.XXXX] which lies within / extends beyond the envelope,
\{\{confirming\,/\,refuting\}\} the mechanism argument."}}

Honest threats to validity for this subsection:

\begin{itemize}
\item The V100 cells use sample-ratio $1.0$ (full data, $\sim 14.5$\,M
  training rows); the 4060~Ti KL-ablation uses sample-ratio $0.1$
  ($\sim 1.45$\,M rows) due to 30~GiB host-RAM constraint. The
  ablation is therefore self-paired (gamma=0 vs gamma=1 on the same
  4060 slice) and not directly compared against the V100 absolute
  AUC values reported in Table~\ref{tab:sam-family-headline}.
\item Both algorithms run under Adam local optimiser ($\eta = 5\mathrm{e}{-4}$),
  while the FedSCAM paper~\cite{FedSCAM2026} and FedGMT
  paper~\cite{FedGMT2025} both use SGD with momentum $= 0.9$.
  Our entire v7 pipeline standardises on Adam (per ADR-001
  D-22 perf-checklist); canonical FedDyn diverged under this
  configuration at 100 rounds (memory
  \verb|project_v5_state.md|, 2026-04-28). The
  \texttt{NonFiniteLossError} guard
  (\verb|_local_loop.py| line 23, added in commit
  \texttt{b8a126e}) isolates any divergent cells; we report any
  NaN exclusions in the paragraph above.
\item FedSCAM's optional K-means clustering and random projection
  $\mathrm{Proj}_d$ steps (paper Algorithm~1 lines marked
  ``optional'') are not implemented in our in-tree port. Our
  $\sim 44$K-parameter LSTM does not benefit from $d \in \{256, 512\}$
  projection (a $176$\,KiB direction vector is cheap regardless), and
  the clustering targets adversarial-client conflict dampening that
  is not a feature of our preregistered partition modes. These
  deferrals are also documented in the algorithm module's
  module-level docstring.
\item FedGMT's Algorithm 1 line 7 places the EMA update at the
  per-client loop start; the reference implementation~\verb|harrylee999/FL-SAM|
  places it at the server's round end. We follow the reference
  (one-round phase shift vs strict paper reading); both share the
  same recurrence $e_t = \alpha e_{t-1} + (1-\alpha) w_t$.
\end{itemize}

\paragraph{Bib entries used (already in \texttt{bibliography.bib}).}

\begin{itemize}
\item \verb|FedSCAM2026|: Rahil, Ahmad, Asif. arXiv:2601.00853 (Jan 2026).
\item \verb|FedGMT2025|: Li, Liu, Cui, Hu, Li. ICML 2025; OpenReview \verb|80mK2Mqaph|.
\end{itemize}

---

## How to fill placeholders

Run the aggregator on both result directories:

```bash
python scripts/aggregate_v7_results.py \
    --sweep-dir artifacts/v7_sam_family \
    --output docs/RESULTS_SAM_FAMILY.md
```

Then read the markdown output's per-cell mean ± std rows and fill in
the placeholders in this document. The KL-ablation paired deltas are
computed directly from the per-seed `summary.json` files in
`artifacts/v7_fedgmt_kl_off/` vs `artifacts/v7_fedgmt_kl_on/` — the
watcher for the paired run already prints the Δ summary.

Once all placeholders are filled, paste into `paper/main.tex` after
line 587 (end of §7.5 `sec:why-no-fedswa`) and rebuild PDF.
