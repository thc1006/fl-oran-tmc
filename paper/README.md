# LaTeX paper build (S11 / JSAC submission)

This directory holds the LaTeX paper that mirrors `docs/PAPER_DRAFT.md`. The Markdown remains the source-of-truth for content; this LaTeX project is the format-compliant version submitted to JSAC.

## Status

In-progress migration; see PR #1 task tracker `S11-A` through `S11-I`.

| Sub-stage | Scope | Status |
|---|---|---|
| S11-A | Skeleton + IEEE template | done (see this README) |
| S11-B | Title + authors + abstract | done (in `main.tex`) |
| S11-C | §1 Introduction | partial — first 2 paragraphs migrated; remainder TODO |
| S11-D | `bibliography.bib` from inline citations | partial — §1 + 9 §9.1 dangling-arxiv entries seeded |
| S11-E | §2–§5 (Related, Dataset, Method, Repro) | not started |
| S11-F | §6–§7 (Results, Discussion) + figures | not started |
| S11-G | §8–§9 (Limitations, Conclusion) | not started |
| S11-H | Supplementary App. A–D | not started |
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
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # second pass resolves cross-refs
```

Output: `paper/main.pdf`.

## Layout

* `main.tex` — top-level LaTeX. `\documentclass[journal]{IEEEtran}` for JSAC regular paper.
* `bibliography.bib` — BibTeX entries; one per `\cite{}` in `main.tex` or `supplementary.tex`. Closes audit Finding B (the 9 previously-dangling arxiv inline refs in §9.1 / App. D now have entries here, even where venue/authors are placeholder pending camera-ready resolution).
* `supplementary.tex` — (S11-H) appendices A–D mirroring `docs/PAPER_SUPPLEMENTARY.md`.

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
