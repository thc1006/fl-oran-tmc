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
| S11-I | Content-equivalence audit + bib batch lift | done (structural 48/48 main + 17/17 supp; 16 numerical sites verified; 11 bib entries enriched with arXiv IDs/venues from PAPER_DRAFT.md References) |

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

## Bibliography TODOs (post S11-I batch lift)

Author/venue placeholders. S11-I batch (commit pending) lifted arXiv IDs and venues from `docs/PAPER_DRAFT.md` References section (L371-421) into bib entries — 11 entries now have arXiv IDs and/or venues that were missing. Remaining placeholders are entries the author's References list also marks as incomplete; they require external lookup at camera-ready.

### Resolved by S11-I batch (lifted from PAPER_DRAFT.md References)

- [x] `Shchur2025_fevbench` — added Shchur first author + arXiv:2509.26468
- [x] `Barker2025_REAL` — added 4-author list + IEEE ICC Workshops 2025 venue
- [x] `Hayek2025` — added arXiv:2504.04678
- [x] `Asperti2021_alphaFLOPs` — added 3-author list + arXiv:2107.11949 + LOD 2021 venue
- [x] `Shen2023_BitBudget` — added arXiv:2311.10802 + CVPR 2024 venue
- [x] `Chung2026_Joules` — added arXiv:2601.22076
- [x] `Spyra2025_BeyondBackprop` — added arXiv:2509.19063 + Spyra first author
- [x] `Li2022_NIIDBench` — added arXiv:2102.02079
- [x] `arXiv2412_17305_FedLEC` — added IJCAI 2025 venue
- [x] `arXiv2602_12009_DPSNNFL` — added ICASSP 2026 venue
- [x] `SpikingPointMamba2025` / `SpikingSSMs2025` — venue confirmed (ICCV 2025 / AAAI 2025); author list TBD note added

### Resolved by camera-ready web batch (WebFetch arxiv abstracts + WebSearch)

- [x] `Groen2023_TRACTOR` — Groen, Yang, Muruganandham, Belgiovine, Ying, Chowdhury + arXiv:2312.07896 + corrected title ("From Classification to Optimization: Slicing and Resource Management with TRACTOR")
- [x] `Chen2024_SpikMamba` — Chen, Yang, Deng, Teng, Pan + arXiv:2410.16746 + 6th ACM MMAsia 2024 venue + corrected subtitle
- [x] `arXiv2408_11823_MambaSpike` — Qin, Liu + CGI 2024 venue
- [x] `arXiv2510_04595_SpikingMamba` — 8-author list + TMLR 2026 venue + corrected subtitle ("Energy-Efficient LLMs via Knowledge Distillation from Mamba")
- [x] `arXiv2509_05276_SpikingBrain` — 19-author SpikingBrain team (Pan/Feng/Zhuang/...Li) + corrected title
- [x] `arXiv2106_06579_FLSNN` — Venkatesha, Kim, Tassiulas, Panda + IEEE Transactions on Signal Processing
- [x] `arXiv2407_17672_VFL_SNN` — Abbasihafshejani, Maiti, Jadliwala + corrected subtitle ("Performance Trade-offs")
- [x] `arXiv2501_03306_RobustSNNFL` — Nguyen, Zhao, Deng, Wu + corrected title ("Against Non-omniscient Byzantine Attacks")
- [x] `arXiv2511_21181_PrivSNNFL` — Aksu, Martinez del Rincon, Alouani
- [x] `pFedFDA2024` — Mclaughlin, Su + NeurIPS 2024 confirmed + arXiv:2411.00329 + corrected title ("Personalized Federated Learning via Feature Distribution Adaptation")
- [x] `FedMoSWA2025` — same paper as Liu2025_FedSWA (Liu et al., ICML 2025, arXiv:2507.20016); FedMoSWA is the momentum-augmented variant introduced in the same paper, kept as separate cite key
- [x] `Wei2018_atan` — replaced "Wei et al. 2018" misattribution with the canonical surrogate-gradient reference: Neftci, Mostafa, Zenke, IEEE Signal Processing Magazine 2019, arXiv:1901.09948 (the actual basis for snntorch's `ATan` surrogate). Cite key retained to avoid main.tex churn; bib note documents the substitution rationale.

### Truly remaining (no arXiv ID; require author-specific external lookup)

- [ ] `Statistical2021_B5G` — generic IEEE 2021 RAN-slicing FL paper. Source markdown lists no arxiv ID and the title is too generic for safe attribution. Author must provide specific reference at camera-ready (the paper exists in §2.2 prose; the citation key needs a definite source).
- [ ] `CDFaware2021` — same situation as Statistical2021_B5G. Author must provide specific IEEE 2021 reference.

(`{(authors)}` placeholder convention from `18640f4` S11-D pass — never strip the placeholder, only fill it. Bib entries for these 2 retain placeholder + descriptive title; all other entries now have verified author lists.)
