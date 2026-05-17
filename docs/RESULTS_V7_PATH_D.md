# Path D — SAM-family multi-arch sweep aggregate

**Generated**: 2026-05-17 22:12:31
**Path D cells discovered**: 100
**Phase 5 baseline cells**: 360

Pairing rule (per paper §2.6):
- `fedscam`, `fedmoswa` → `fedavg` (FedAvg-class + extra step)
- `fedgmt` → `fedadam` (adaptive)

Significance: ★ = paired-bootstrap CI95 excludes 0 positively; ↓ = CI95 excludes 0 negatively; — = CI95 straddles 0; `prelim` = n < 3 paired seeds (CI not computed).

## LSTM

| Algo | Partition | n paired | Path D AUC | Baseline AUC | Δ AUC | 95% CI | Wilcoxon p | sig |
|---|---|---:|---|---|---:|---|---:|---|
| FedGMT | IID | 7 | 0.9124 ± 0.0010 | 0.9178 ± 0.0005 | -0.0053 | [-0.0062, -0.0045] | 0.0156 | ↓ 顯著劣 |
| FedMoSWA | IID | 0 | — | 0.9159 ± 0.0004 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | IID | 10 | 0.9162 ± 0.0004 | 0.9159 ± 0.0004 | +0.0003 | [+0.0000, +0.0005] | 0.0840 | ★ 顯著優 |
| FedGMT | Dir α=0.05 | 7 | 0.8431 ± 0.0238 | 0.8653 ± 0.0184 | -0.0233 | [-0.0327, -0.0147] | 0.0156 | ↓ 顯著劣 |
| FedMoSWA | Dir α=0.05 | 0 | — | 0.8605 ± 0.0161 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=0.05 | 10 | 0.8635 ± 0.0170 | 0.8605 ± 0.0161 | +0.0030 | [+0.0005, +0.0054] | 0.0840 | ★ 顯著優 |
| FedGMT | Dir α=0.1 | 7 | 0.8163 ± 0.0298 | 0.8489 ± 0.0241 | -0.0274 | [-0.0368, -0.0184] | 0.0156 | ↓ 顯著劣 |
| FedMoSWA | Dir α=0.1 | 0 | — | 0.8361 ± 0.0302 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=0.1 | 10 | 0.8431 ± 0.0273 | 0.8361 ± 0.0302 | +0.0070 | [+0.0031, +0.0121] | 0.0020 | ★ 顯著優 |
| FedGMT | Dir α=0.5 | 5 | 0.7740 ± 0.0319 | 0.7864 ± 0.0211 | -0.0145 | [-0.0190, -0.0104] | 0.0625 | ↓ 顯著劣 |
| FedMoSWA | Dir α=0.5 | 0 | — | 0.7794 ± 0.0223 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=0.5 | 10 | 0.7824 ± 0.0186 | 0.7794 ± 0.0223 | +0.0030 | [-0.0000, +0.0059] | 0.0488 | — 無顯著 |
| FedGMT | Dir α=1 | 5 | 0.7454 ± 0.0022 | 0.7650 ± 0.0084 | -0.0180 | [-0.0242, -0.0119] | 0.0625 | ↓ 顯著劣 |
| FedMoSWA | Dir α=1 | 0 | — | 0.7571 ± 0.0061 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=1 | 10 | 0.7583 ± 0.0069 | 0.7571 ± 0.0061 | +0.0012 | [+0.0001, +0.0025] | 0.0645 | ★ 顯著優 |
| FedGMT | Dir α=5 | 5 | 0.7418 ± 0.0044 | 0.7552 ± 0.0040 | -0.0109 | [-0.0158, -0.0075] | 0.0625 | ↓ 顯著劣 |
| FedMoSWA | Dir α=5 | 0 | — | 0.7475 ± 0.0044 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=5 | 10 | 0.7465 ± 0.0023 | 0.7475 ± 0.0044 | -0.0011 | [-0.0031, +0.0007] | 0.4316 | — 無顯著 |

## Mamba

| Algo | Partition | n paired | Path D AUC | Baseline AUC | Δ AUC | 95% CI | Wilcoxon p | sig |
|---|---|---:|---|---|---:|---|---:|---|
| FedGMT | IID | 0 | — | 0.9186 ± 0.0004 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | IID | 1 | 0.9016 | 0.9165 ± 0.0006 | -0.0146 | — | — | prelim n=1 |
| FedSCAM | IID | 0 | — | 0.9165 ± 0.0006 | +0.0000 | — | — | prelim n=0 |
| FedGMT | Dir α=0.05 | 0 | — | 0.8665 ± 0.0188 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=0.05 | 0 | — | 0.8686 ± 0.0143 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=0.05 | 1 | 0.8646 | 0.8686 ± 0.0143 | -0.0027 | — | — | prelim n=1 |
| FedGMT | Dir α=0.1 | 0 | — | 0.8579 ± 0.0260 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=0.1 | 0 | — | 0.8524 ± 0.0251 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=0.1 | 0 | — | 0.8524 ± 0.0251 | +0.0000 | — | — | prelim n=0 |
| FedGMT | Dir α=0.5 | 0 | — | 0.7913 ± 0.0229 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=0.5 | 0 | — | 0.7816 ± 0.0225 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=0.5 | 0 | — | 0.7816 ± 0.0225 | +0.0000 | — | — | prelim n=0 |
| FedGMT | Dir α=1 | 0 | — | 0.7656 ± 0.0119 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=1 | 0 | — | 0.7561 ± 0.0091 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=1 | 0 | — | 0.7561 ± 0.0091 | +0.0000 | — | — | prelim n=0 |
| FedGMT | Dir α=5 | 0 | — | 0.7574 ± 0.0074 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=5 | 0 | — | 0.7490 ± 0.0073 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=5 | 0 | — | 0.7490 ± 0.0073 | +0.0000 | — | — | prelim n=0 |

## Spiking

| Algo | Partition | n paired | Path D AUC | Baseline AUC | Δ AUC | 95% CI | Wilcoxon p | sig |
|---|---|---:|---|---|---:|---|---:|---|
| FedGMT | IID | 0 | — | 0.8563 ± 0.0118 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | IID | 1 | 0.8325 | 0.8529 ± 0.0051 | -0.0213 | — | — | prelim n=1 |
| FedSCAM | IID | 0 | — | 0.8529 ± 0.0051 | +0.0000 | — | — | prelim n=0 |
| FedGMT | Dir α=0.05 | 0 | — | 0.7080 ± 0.0493 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=0.05 | 0 | — | 0.6914 ± 0.0477 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=0.05 | 1 | 0.6614 | 0.6914 ± 0.0477 | -0.0007 | — | — | prelim n=1 |
| FedGMT | Dir α=0.1 | 0 | — | 0.7052 ± 0.0418 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=0.1 | 0 | — | 0.6785 ± 0.0133 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=0.1 | 0 | — | 0.6785 ± 0.0133 | +0.0000 | — | — | prelim n=0 |
| FedGMT | Dir α=0.5 | 0 | — | 0.6883 ± 0.0117 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=0.5 | 0 | — | 0.6735 ± 0.0041 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=0.5 | 0 | — | 0.6735 ± 0.0041 | +0.0000 | — | — | prelim n=0 |
| FedGMT | Dir α=1 | 0 | — | 0.6823 ± 0.0064 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=1 | 0 | — | 0.6694 ± 0.0025 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=1 | 0 | — | 0.6694 ± 0.0025 | +0.0000 | — | — | prelim n=0 |
| FedGMT | Dir α=5 | 0 | — | 0.6797 ± 0.0072 | +0.0000 | — | — | prelim n=0 |
| FedMoSWA | Dir α=5 | 0 | — | 0.6689 ± 0.0031 | +0.0000 | — | — | prelim n=0 |
| FedSCAM | Dir α=5 | 0 | — | 0.6689 ± 0.0031 | +0.0000 | — | — | prelim n=0 |

## Aggregate verdict

- Significant wins (★): **4**
- Significant losses (↓): **6**
- No significant difference (—): **2**
- Preliminary (n<3 paired): **42**

