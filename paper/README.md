# LaTeX paper build (S11 / JSAC submission)

This directory holds the LaTeX paper that mirrors `docs/PAPER_DRAFT.md`. The Markdown remains the source-of-truth for content; this LaTeX project is the format-compliant version submitted to JSAC.

## Status

In-progress migration; see PR #1 task tracker `S11-A` through `S11-I`.

| Sub-stage | Scope | Status |
|---|---|---|
| S11-A | Skeleton + IEEE template | done (see this README) |
| S11-B | Title + authors + abstract | done (in `main.tex`) |
| S11-C | §1 Introduction | done (commit `a3c5eac`) |
| S11-D | `bibliography.bib` from inline citations | done (61/61 entries cited after S11-G §9.1 wired the 9 dangling-arxiv stubs; supp may add more in S11-H) |
| S11-E | §2–§5 (Related, Dataset, Method, Repro) | done (commits `a69a3a1` §2, `30a9840` §3, `087242b` §4, `7e15d3c` §5) |
| S11-F | §6–§7 (Results, Discussion) + figures | done (commits `c02843f` §6+3 figures, `6777b68` §7+2 tables) |
| S11-G | §8–§9 (Limitations, Conclusion) | done (commits `6495f8f` §8+14 bullets, `3718a8d` §9+9 arxiv-cite wires) |
| S11-H | Supplementary App. A–D | done (`supplementary.tex` standalone IEEEtran, 3pp PDF, 12 \cite{} all resolve) |
| S11-I | Content-equivalence audit | not started |

## Build

LaTeX toolchain not installed in the dev environment. Install once:

```bash
sudo apt-get install -y \
    texlive-latex-base \
    texlive-publishers \
    texlive-bibtex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra
```

Then build:

```bash
cd paper
# Main paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # second pass resolves cross-refs

# Supplementary (independent build, shares bibliography.bib)
pdflatex supplementary.tex
bibtex supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex
```

Outputs: `paper/main.pdf` (16pp), `paper/supplementary.pdf` (3pp).

## Layout

* `main.tex` — top-level LaTeX. `\documentclass[journal]{IEEEtran}` for JSAC regular paper.
* `bibliography.bib` — BibTeX entries; one per `\cite{}` in `main.tex` or `supplementary.tex`. Closes audit Finding B (the 9 previously-dangling arxiv inline refs in §9.1 / App. D now have entries here, even where venue/authors are placeholder pending camera-ready resolution).
* `supplementary.tex` — appendices A–D mirroring `docs/PAPER_SUPPLEMENTARY.md`. Standalone IEEEtran document (independent compile, own page numbering); shares `bibliography.bib`. Cross-paper refs to main use `\S\,X.Y` literal form (cannot `\ref` across documents).

## Citation key convention

`@article{FirstAuthorYear[_ShortTitle], ...}` — example: `Polese2022`, `Caldas2018_LEAF`, `Shen2025_STEP`. For arxiv-only mentions where authors are not finalised (the 9 §9.1 dangling refs), keep the placeholder `{(authors)}` in the `author` field and resolve at camera-ready.

## Cross-reference convention

* Sections: `\label{sec:intro}` / `\ref{sec:intro}` (mirrors Markdown §1).
* Equations: `\label{eq:fedavg-update}` / `\eqref{eq:fedavg-update}`.
* Figures: `\label{fig:pareto}` / `\ref{fig:pareto}`.
* Tables: `\label{tab:results-stage2}` / `\ref{tab:results-stage2}`.

## Migration notes

* Markdown bold `**foo**` → `\textbf{foo}`; italic `*foo*` → `\emph{foo}`.
* Markdown inline backticks → `\texttt{...}` for prose; `\lstinline|...|` for longer code.
* Greek letters in inline numerics: `\alpha`, `\Delta`, `\sigma`, `\beta_1`, etc.
* `≈ ± × ≥ ≤ →` → `\approx \pm \times \geq \leq \rightarrow`.
* Figure inputs from `artifacts/figures/*.png` via relative path: `\includegraphics[width=\columnwidth]{../artifacts/figures/pareto.png}`.

## Bibliography TODOs (S11-E pass)

Author/venue placeholders introduced during the §2 migration. Each entry has a `{(authors)}` author block or generic `(venue TBD)` slot that must be resolved before camera-ready. Resolve in one batch after S11-E + S11-F + S11-G complete to avoid churn.

- [ ] `Shchur2025_fevbench` — fev-bench paper authors + venue (NeurIPS 2025?)
- [ ] `Statistical2021_B5G` — exact IEEE conf/journal + author list
- [ ] `CDFaware2021` — IEEE venue + author list
- [ ] `Groen2023_TRACTOR` — full author list + IEEE venue (likely INFOCOM/Globecom)
- [ ] `Hayek2025` — full author list + venue
- [ ] `Chen2024_SpikMamba` — venue (NeurIPS 2024?) + co-authors
- [ ] `arXiv2408_11823_MambaSpike` — author list
- [ ] `SpikingPointMamba2025` — confirm ICCV 2025 + author list
- [ ] `SpikingSSMs2025` — confirm AAAI 2025 + author list
- [ ] `arXiv2510_04595_SpikingMamba` — author list
- [ ] `arXiv2509_05276_SpikingBrain` — author list
- [ ] `arXiv2106_06579_FLSNN` — author list
- [ ] `arXiv2412_17305_FedLEC` — author list
- [ ] `arXiv2407_17672_VFL_SNN` — author list
- [ ] `arXiv2501_03306_RobustSNNFL` — author list
- [ ] `arXiv2511_21181_PrivSNNFL` — author list
- [ ] `arXiv2602_12009_DPSNNFL` — author list
- [ ] `Shen2023_BitBudget` — venue + co-authors
- [ ] `Asperti2021_alphaFLOPs` — venue + co-authors
- [ ] `Chung2026_Joules` — venue + arXiv ID
- [ ] `Spyra2025_BeyondBackprop` — venue + co-authors
- [ ] `pFedFDA2024` — confirm NeurIPS 2024 + author list
- [ ] `Wei2018_atan` — exact paper for the atan surrogate gradient (markdown source says "Wei et al. 2018"; common candidates are Wu et al. 2018 STBP or Neftci et al. 2019; verify before camera-ready)
- [ ] `FedMoSWA2025` — exact author list + venue (markdown §7.5 lists it alongside FedSWA + FedSCAM; arXiv ID TBD)

(`{(authors)}` placeholder convention from `18640f4` S11-D pass — never strip the placeholder, only fill it.)
