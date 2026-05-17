# Paper §3 Dataset Description Revision

**Status**: ready to paste into `paper/main.tex` after V100 Path D sweep completes.
**Motivation**: empirical verification of authoritative ColO-RAN GitHub README +
on-disk parquet (2026-05-18) revealed three precision issues in earlier drafts:

1. The target is `rx_errors_ul (%)` (uplink retransmission error rate), NOT BLER.
   Original ColO-RAN CSV has no BLER column; our `ul_bler` is a rescaling
   `rx_errors_ul% / 100`, bit-identical to the source.
2. TTI / sampling interval is **250 ms**, not 1 ms or 1 s. Per-run timestamp
   diffs cluster at 249-251 ms; `seq_len=5` covers 1.25 s context.
3. `dl_bler` and `tx_errors_dl%` are exactly zero everywhere — the Colosseum
   emulator does not inject downlink errors in this scenario. Only uplink SLA
   prediction is supportable.

All three are honest scope statements, not weaknesses.

---

## Drop-in §3 paragraph (paper/main.tex)

```latex
\subsection{Dataset and prediction task}\label{sec:dataset}

We use the Colosseum ColO-RAN dataset~\cite{Polese2022ColORAN}, which emulates
a 5G network with 7 base stations (BSs) and 42 user equipments (UEs) in the
Rome dense-urban scenario (0.11~km$^2$, BS coordinates from
OpenCelliD)~\cite{Polese2022ColORAN}. Each BS serves 3 slices: slice~0
(eMBB, 4~Mbps constant-bitrate per UE), slice~1 (MTC, 30~pkt/s Poisson of
125\,B packets), and slice~2 (URLLC, 10~pkt/s Poisson of 125\,B packets), with
14 UEs assigned to each slice. The dataset enumerates 28 \emph{training
configurations} (\verb|tr0|--\verb|tr27|) differing in resource-block-group
(RBG) allocations across slices, 11 experiment repetitions per configuration,
and 3 scheduling policies (\verb|sched0|: round-robin, \verb|sched1|:
waterfilling, \verb|sched2|: proportionally fair). The 17 continuous KPI
features per row include uplink/downlink MCS, SINR, throughput
(\verb|tx_brate_dl_Mbps|, \verb|rx_brate_ul_Mbps|), buffer occupancy, channel
quality indicator (CQI), power headroom (PHR), granted/requested PRBs, and
uplink turbo-decoder iteration count. We do not use the per-UE CSVs; instead
we consume the per-(BS,~slice) aggregated metrics with timestamps quantised
to the emulator's 250\,ms reporting interval (TTI), giving
$\sim$2{,}100 250\,ms windows per (\verb|tr|, \verb|exp|, \verb|sched|, BS,
slice) tuple.

\textbf{Target.} Given the previous 5 250\,ms windows (1.25\,s) of KPIs for a
specific (run, slice) pair, we predict whether the next window's uplink
retransmission error rate \verb|rx_errors_ul| exceeds 10\,\%. The 10\,\%
threshold corresponds to the 3GPP first-transmission BLER target
(TS\,36.213~\cite{3GPP_TS_36_213}); a window above this threshold indicates
elevated HARQ retransmission load and short-term uplink SLA degradation, the
class of event a near-real-time RIC xApp would intervene
on~\cite{Polese2022ColORAN}. The class is 30.76\,\% positive across the
18.35\,M-row corpus. Sequences are constructed strictly within each
\verb|(run_id, slice_id)| group; consecutive 250\,ms windows for different
slices are not concatenated.

\textbf{Federated partitioning.} The 7 BSs serve as the natural FL clients —
a realistic operator scenario where each BS holds its local KPI stream. The
28 \verb|tr| configurations are NOT used as clients; they are an internal
experimental variable affecting RBG allocation and are treated as part of
the within-client data distribution. Inter-client heterogeneity arises from
(i) channel realisations differing across the 7 Rome locations and (ii) UE
traffic mix per BS. Dirichlet partitioning with concentration
$\alpha \in \{0.05, 0.1, 0.5, 1.0, 5.0\}$ plus IID provides controlled
heterogeneity sweeps on top of the natural inter-BS variation.

\textbf{Scope.} The Colosseum emulator does not inject downlink errors in
this scenario: \verb|dl_bler| and \verb|tx_errors_dl%| are uniformly zero
across all 18.35\,M rows. Consequently we evaluate only uplink SLA
prediction. Three further dataset columns (\verb|ul_rssi|, \verb|dl_pmi|,
\verb|dl_ri|) are also fixed at zero and are dropped from the feature set.
We make no claims about downlink SLA prediction on this corpus.
```

---

## Drop-in §future-work paragraph (paper/main.tex)

```latex
\paragraph{Beyond binary uplink-SLA classification.}
Our benchmark restricts attention to 250\,ms-ahead binary uplink-SLA
prediction. Four natural extensions, each preserving the FL setting we
established, would broaden the empirical picture:

\begin{itemize}
  \item \textbf{Continuous regression on \texttt{rx\_errors\_ul\%}}: $>$50\,\%
    of windows have exactly zero uplink retransmission, motivating a
    zero-inflated mixture head (binary $0$-vs-positive plus conditional
    Beta-regression on the positive support). This recovers magnitude
    information that the $>$10\,\% binarisation discards.

  \item \textbf{Multi-step ahead forecasting}: predicting the next $K$
    250\,ms windows (instead of just $K{=}1$) matches longer xApp action
    cycles and exercises the longer-context capacity of Mamba and
    Spiking-SSM backbones that the 1.25\,s history under-utilises.

  \item \textbf{Slice-aware multi-task prediction}: jointly forecasting
    eMBB downlink throughput (\texttt{tx\_brate\_dl\_Mbps}), URLLC uplink
    retransmission rate, and MTC error rate as three slice-specific SLA
    metrics, each with its own threshold (4\,Mbps shortfall for eMBB,
    10\,\% retransmission for URLLC/MTC). This addresses the per-slice
    operational metrics that a slice-aware operator would actually
    monitor.

  \item \textbf{Cross-traffic-configuration generalisation}: training on a
    subset of the 28 RBG-allocation configurations (\texttt{tr0}--\texttt{tr27})
    and testing on a held-out subset, evaluating whether algorithms learn
    transferable structure rather than memorising configuration-specific
    KPIs.
\end{itemize}

We also note that the Colosseum emulator's lack of injected downlink errors
means a complete uplink--downlink SLA evaluation requires a different
testbed (e.g., commercial channel traces twinned to Colosseum per the
methodology of~\cite{Bonati2024Twinning}).
```

---

## Optional §threats-to-validity addition

```latex
\paragraph{Dataset scope and target proxy.}
The Colosseum ColO-RAN dataset's \verb|tx_errors_dl%| field is uniformly
zero across all 18.35\,M rows we use, reflecting the emulator's lack of
downlink error injection in this scenario. We therefore predict only
uplink SLA violations, parameterised as \verb|rx_errors_ul% > 10|. While
this 10\,\% threshold is grounded in 3GPP TS\,36.213's first-transmission
BLER target~\cite{3GPP_TS_36_213}, it is a proxy for the operationally
richer event of HARQ-after BLER exceeding $10^{-3}$, which the dataset
does not expose directly. Conclusions drawn on this proxy should be
re-examined on testbeds with explicit downlink error injection or
HARQ-tracking traces.
```

---

## BibTeX entries to add (paper/bibliography.bib)

```bibtex
@article{Polese2022ColORAN,
  author    = {Polese, Michele and Bonati, Leonardo and D'Oro, Salvatore and
               Basagni, Stefano and Melodia, Tommaso},
  title     = {ColO-RAN: Developing Machine Learning-Based xApps for
               {O}pen {RAN} Closed-Loop Control on Programmable
               Experimental Platforms},
  journal   = {IEEE Transactions on Mobile Computing},
  year      = {2022},
  doi       = {10.1109/TMC.2022.3188013},
  note      = {Dataset available at
               \url{https://github.com/wineslab/colosseum-oran-coloran-dataset}},
}

@techreport{3GPP_TS_36_213,
  author      = {{3GPP}},
  title       = {Evolved Universal Terrestrial Radio Access ({E-UTRA});
                 Physical layer procedures},
  institution = {3rd Generation Partnership Project},
  number      = {TS 36.213},
  year        = {2024},
  note        = {Section 9.1 specifies first-transmission BLER target
                 $\leq 10\%$ for link adaptation},
}

@inproceedings{Bonati2024Twinning,
  author    = {Bonati, Leonardo and others},
  title     = {Twinning Commercial Network Traces on Experimental
               {O}pen {RAN} Platforms},
  booktitle = {Proc.\ ACM MobiCom},
  year      = {2024},
  doi       = {10.1145/3636534.3697320},
}
```

---

## Reviewer-anticipation notes (do NOT paste, internal use)

A JSAC reviewer might ask:

- **Q**: "Why 10\,\%? Is this just chosen post-hoc to balance classes?"
  - A: 30.76\,\% positive rate is incidental; threshold comes from 3GPP
    TS 36.213. We explicitly cite the standard.

- **Q**: "Why not predict BLER directly as continuous?"
  - A: Dataset has no BLER column. We derive uplink retransmission rate
    from `rx_errors_ul%`. Continuous regression is listed in future work.

- **Q**: "Why only 7 clients? Real operator scenarios have hundreds."
  - A: 7 corresponds to the dataset's 7 BSs in Rome dense-urban; we are
    constrained by the public benchmark. Larger-scale FL is a separate
    deployment concern. We test 6 Dirichlet $\alpha$ partitions to surface
    heterogeneity beyond the natural inter-BS variation.

- **Q**: "What is the TTI? You vaguely say 'time step'."
  - A: 250\,ms, the Colosseum reporting interval. Our `seq_len=5` covers
    1.25\,s context, matching near-RT RIC action cycles per
    O-RAN.WG3.E2GAP-v04.00.
