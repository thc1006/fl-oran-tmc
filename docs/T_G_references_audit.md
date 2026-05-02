# Reference Audit — PAPER_DRAFT.md §2 (Related Work)

Date: 2026-05-01
Auditor: thc1006 (via Claude verification pipeline)
Scope: 24 priority citations across §2.1–§2.6
Method: WebFetch arxiv.org/abs/<id> + Google Scholar / web search; max 5 min/citation, 90 min hard cap
Result: 24/24 verified within budget. Zero hallucinations. Two minor metadata refinements suggested.

---

## Verified (correct as cited)

### §2.4 GPU energy measurement methodology

1. **ML.ENERGY Benchmark [Chung et al. 2025]** — confirmed via https://arxiv.org/abs/2505.06371
   - Title: "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization"
   - Authors: Jae-Won Chung, Jeff J. Ma, Ruofan Wu, Jiachen Liu, Oh Jun Kweon, Yuxuan Xia, Zhiyu Wu, Mosharaf Chowdhury
   - First submitted May 9, 2025; v2 Oct 16, 2025
   - **Important:** This IS a real arXiv paper, not just a blog. The paper text uses the leaderboard website as a companion artifact. Drafts §2.4 sentence "ML.ENERGY 2025 best-practices blog" should be tightened to "ML.ENERGY 2025 paper" (or "paper and accompanying leaderboard") to avoid downgrading the citation.

2. **Where Do the Joules Go? [Chung et al. 2026]** — confirmed via https://arxiv.org/abs/2601.22076
   - Title: "Where Do the Joules Go? Diagnosing Inference Energy Consumption"
   - Lead author: Jae-Won Chung et al.
   - arXiv submission Jan 29, 2026. Measures 46 models / 7 tasks / 1858 configs on H100 + B200 GPUs.

3. **STEP [Shen et al. 2025]** — confirmed via https://arxiv.org/abs/2505.11151
   - Title: "STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking"
   - arXiv May 16, 2025. Includes unified analytical energy model (spike sparsity, bitwidth, memory access). Matches paper claim about benchmarking spiking transformers including memory-access cost.

4. **Beyond Backpropagation [Spyra 2025]** — confirmed via https://arxiv.org/abs/2509.19063
   - Title: "Beyond Backpropagation: Exploring Innovative Algorithms for Energy-Efficient Deep Neural Network Training"
   - Author: Przemysław Spyra (single author; **"Spyra" is a real Polish surname**, suspicion unfounded)
   - arXiv Sep 23, 2025. Uses NVML + CodeCarbon exactly as cited.

5. **α-FLOPs [Asperti et al. 2021]** — confirmed via https://arxiv.org/abs/2107.11949
   - Actual published title: "Dissecting FLOPs along input dimensions for GreenAI cost estimations"
   - Authors: Andrea Asperti (Bologna, Italian), Davide Evangelista, Moreno Marzolla
   - Venue: 7th International Conference on Machine Learning, Optimization and Data Science (LOD 2021)
   - The "α-FLOPs" name is the technique introduced inside the paper. Citation acceptable as-is; consider also citing LOD 2021 venue name explicitly for peer-reviewed status.

6. **Shen et al. 2023 "Is Conventional SNN Really Efficient?"** — confirmed via https://arxiv.org/abs/2311.10802
   - Full title: "Is Conventional SNN Really Efficient? A Perspective from Network Quantization"
   - arXiv Nov 17, 2023. Bit Budget framework as described. **Also published at CVPR 2024** (https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Are_Conventional_SNNs_...). Citing CVPR 2024 (not just arXiv 2023) would strengthen peer-review credit; current citation is technically correct but undersells the paper.

### §2.1 Federated learning benchmarks

7. **fev-bench [Shchur et al. 2025]** — confirmed via https://arxiv.org/abs/2509.26468
   - Title: "fev-bench: A Realistic Benchmark for Time Series Forecasting"
   - Lead author: Oleksandr Shchur (+7 authors). arXiv Sep 30, 2025; rev Feb 3, 2026.
   - Exactly matches "rigorous bootstrap-CI95 evaluation for time-series forecasting" claim.

8. **pfl-research [Granqvist et al. 2024]** — confirmed via https://arxiv.org/abs/2404.06430
   - Title: "pfl-research: simulation framework for accelerating research in Private Federated Learning"
   - Lead author: Filip Granqvist (Apple) + Congzheng Song, Áine Cahill, Rogier van Dalen, et al.
   - arXiv Apr 9, 2024; v2 Dec 10, 2024. **Published in NeurIPS 2024 Datasets and Benchmarks Track** (https://proceedings.neurips.cc/paper_files/paper/2024/...). Consider citing NeurIPS 2024 venue for full credit.

### §2.2 FL on cellular and wireless networks

9. **Hayek et al. 2025** — confirmed via https://arxiv.org/abs/2504.04678
   - Title: "Federated Learning over 5G, WiFi, and Ethernet: Measurements and Evaluation"
   - 5G-NR Standalone testbed + Raspberry Pi clients + O-RAN central server + Flower FL framework. Matches paper description exactly.

10. **arxiv 2508.08479 [2025]** — confirmed via https://arxiv.org/abs/2508.08479
    - Title: "Benchmarking Federated Learning for Throughput Prediction in 5G Live Streaming Applications"
    - Authors: Yuvraj Dutta, Soumyajit Chatterjee, Sandip Chakraborty, Basabdatta Palit
    - Submitted Aug 11, 2025 (IEEE TNET submission). Benchmarks FedAvg/FedProx/FedBN across LSTM/CNN/CNN+LSTM/Transformer on 5 throughput-prediction datasets. FedBN superior under non-IID confirmed. **Note: paper actually uses LSTM/CNN/CNN+LSTM/Transformer, not LSTM/CNN/Transformer as cited** — minor refinement: add "CNN+LSTM" to the listed architectures.

### §2.3 Spiking-SSM and FL × spiking neural networks

11. **SpikingMamba [arxiv 2510.04595, 2025]** — confirmed via https://arxiv.org/abs/2510.04595
    - Title: "SpikingMamba: Towards Energy-Efficient Large Language Models via Knowledge Distillation from Mamba"
    - Authors: Yulong Huang, Jianxiong Tang, Chao Wang, Ziyi Wang, Jianguo Zhang, Zhichao Lu, Bojun Cheng, Luziwei Leng
    - Submitted Oct 6, 2025; rev Apr 12, 2026. Distills LLM (Mamba) into spiking variant. Matches.

12. **SpikingBrain [arxiv 2509.05276, 2025]** — confirmed via https://arxiv.org/abs/2509.05276
    - Title: "SpikingBrain: Spiking Brain-inspired Large Models" (19 authors led by Yuqi Pan; senior author Guoqi Li)
    - Submitted Sep 5, 2025; rev Dec 1, 2025. Includes SpikingBrain-7B (linear) and SpikingBrain-76B (hybrid-linear MoE). 76B parameter claim verified.

13. **arxiv 2106.06579 founding FL × SNN [2021]** — confirmed via https://arxiv.org/abs/2106.06579
    - Title: "Federated Learning with Spiking Neural Networks"
    - Authors: Yeshwanth Venkatesha, Youngeun Kim, Leandros Tassiulas, Priyadarshini Panda
    - Submitted Jun 11, 2021. Reports up to 5.3× energy efficiency. First FL+SNN paper, matches "founding" claim.

14. **FedLEC [arxiv 2412.17305, 2024]** — confirmed via https://arxiv.org/abs/2412.17305
    - Title: "Exploiting Label Skewness for Spiking Neural Networks in Federated Learning"
    - Authors: Di Yu, Xin Du, Linshan Jiang, Huijing Zhang, Shuiguang Deng
    - Submitted Dec 23, 2024; **accepted to IJCAI 2025**. Consider adding IJCAI 2025 venue.

15. **SNN in Vertical FL [arxiv 2407.17672, 2024]** — confirmed via https://arxiv.org/abs/2407.17672
    - Title: "Spiking Neural Networks in Vertical Federated Learning: Performance Trade-offs"
    - Authors: Maryam Abbasihafshejani, Anindya Maiti, Murtuza Jadliwala. Submitted Jul 24, 2024. Matches.

16. **Robustness of SNN in FL with Compression [arxiv 2501.03306, 2025]** — confirmed via https://arxiv.org/abs/2501.03306
    - Title: "The Robustness of Spiking Neural Networks in Federated Learning with Compression Against Non-omniscient Byzantine Attacks"
    - Authors: Manh V. Nguyen, Liang Zhao, Bobin Deng, Shaoen Wu. Submitted Jan 6, 2025. Matches "Byzantine attacks" claim.

17. **Privacy in FL with SNN [arxiv 2511.21181, 2025]** — confirmed via https://arxiv.org/abs/2511.21181
    - Title: "Privacy in Federated Learning with Spiking Neural Networks"
    - Authors: Dogukan Aksu, Jesus Martinez del Rincon, Ihsen Alouani. Submitted Nov 26, 2025. First systematic gradient-inversion benchmark for spiking architectures. Matches "gradient leakage" claim.

18. **arxiv 2602.12009 firing-rate sensitivity to DP [Feb 2026]** — confirmed via https://arxiv.org/abs/2602.12009
    - Title: "On the Sensitivity of Firing Rate-Based Federated Spiking Neural Networks to Differential Privacy"
    - Authors: Luiz Pereira, Mirko Perkusich, Dalton Valadares, Kyller Gorgônio
    - Submitted Feb 12, 2026. **To appear at IEEE ICASSP 2026.** Consider adding ICASSP 2026 venue. Matches DP-SNN-FL preemption claim.

### §2.5 Heterogeneity in federated learning

19. **Borazjani et al. 2025 [arxiv 2503.14553]** — confirmed via https://arxiv.org/abs/2503.14553
    - Title: "Redefining non-IID Data in Federated Learning for Computer Vision Tasks: Migrating from Labels to Embeddings for Task-Specific Data Distributions"
    - Authors: Kasra Borazjani, Payam Abdisarabshali, Naji Khosravan, Seyyedali Hosseinalipour
    - Submitted Mar 17, 2025; **accepted IEEE Transactions on Artificial Intelligence 2026**. Matches embedding-based heterogeneity claim.

22. **NIID-Bench [Li et al. 2022]** — confirmed via https://arxiv.org/abs/2102.02079 + GitHub Xtra-Computing/NIID-Bench
    - Title: "Federated Learning on Non-IID Data Silos: An Experimental Study"
    - Authors: Qinbin Li, Yiqun Diao, Quan Chen, Bingsheng He
    - **Venue: ICDE 2022** (not generic "2022"). Consider citing ICDE 2022 explicitly.

23. **FedBN [Li et al. ICLR 2021]** — confirmed via https://arxiv.org/abs/2102.07623 + https://openreview.net/forum?id=6YEQUn0QICG
    - Title: "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization"
    - Authors: Xiaoxiao Li, Meirui Jiang, Xiaofei Zhang, Michael Kamp, Qi Dou
    - Venue: ICLR 2021. Citation accurate.

24. **pFedFDA [NeurIPS 2024]** — confirmed via https://proceedings.neurips.cc/paper_files/paper/2024/file/8ce6c5450ccddbe6adee4b3749893587-Paper-Conference.pdf + https://arxiv.org/abs/2411.00329
    - Title: "Personalized Federated Learning via Feature Distribution Adaptation"
    - Authors: Connor J. McLaughlin, Lili Su (Northeastern). NeurIPS 2024 Main Conference Poster.

### §2.6 Sharpness-aware federated learning

20. **Caldarola et al. 2022 (FedSAM)** — confirmed via https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830636.pdf + https://github.com/debcaldarola/fedsam
    - Title: "Improving Generalization in Federated Learning by Seeking Flat Minima"
    - Authors: Debora Caldarola, Barbara Caputo, Marco Ciccone
    - Venue: ECCV 2022. Combines SAM/ASAM client-local + SWA server-side. Matches "FedSAM" descriptor.

21. **FedSCAM** — confirmed via https://arxiv.org/abs/2601.00853
    - Title: "FedSCAM (Federated Sharpness-Aware Minimization with Clustered Aggregation and Modulation): Scam-resistant SAM for Robust Federated Optimization in Heterogeneous Environments"
    - Lead author: Sameer Rahil et al. arXiv Jan 2026 (preprint, no peer-reviewed venue yet)
    - **The paper draft says "FedSCAM [Dec 2025]"; arXiv timestamp is 2601.* which is January 2026, not December 2025**. Minor date refinement: change "[Dec 2025]" → "[Jan 2026, arxiv 2601.00853]". Also note the citation in paper is incomplete — full proper venue is arXiv preprint (no conference acceptance found as of audit date).

— Bonus check —

**FedSWA [Liu et al., ICML 2025; arxiv:2507.20016]** (§2.6) — confirmed via https://arxiv.org/abs/2507.20016
   - Title: "FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging"
   - Authors: Junkang Liu, Yuanyuan Liu, Fanhua Shang, Hongying Liu, Jin Liu, Wei Feng
   - Venue: ICML 2025. Citation accurate.

---

## Corrected (paper has wrong metadata)

None of the 24 priority citations is actually wrong, but the following minor refinements would strengthen the paper:

1. **§2.2 arxiv 2508.08479** — paper actually uses 4 architectures (LSTM, CNN, CNN+LSTM, Transformer), not 3. Add "CNN+LSTM" to the listed-architectures sentence, or generalize as "LSTM/CNN/Transformer-family architectures (incl. CNN+LSTM hybrid)".

2. **§2.4 ML.ENERGY** — paper draft characterizes it as a "blog" in §2.4 paragraph 2 ("the ML.ENERGY 2025 best-practices blog"). It is in fact a published arXiv paper (2505.06371). Tighten wording to "the ML.ENERGY 2025 paper" or "ML.ENERGY 2025 paper and accompanying leaderboard documentation".

3. **§2.6 FedSCAM date** — paper draft says "[Dec 2025]". arXiv ID 2601.00853 is January 2026 timestamp (arXiv prefix YYMM = 2601). Change to "[arxiv 2601.00853, Jan 2026]".

4. **Optional venue upgrades** (current citations technically correct, but adding peer-reviewed venue strengthens credibility):
   - Shen 2023 SNN-Bit-Budget → also CVPR 2024
   - pfl-research → NeurIPS 2024 D&B Track
   - FedLEC → IJCAI 2025
   - 2602.12009 firing-rate-DP → ICASSP 2026
   - Borazjani 2025 → IEEE TAI 2026
   - NIID-Bench → ICDE 2022 (not just "2022")

---

## Hallucinated / not found (cannot verify in 2 search attempts)

**None.** All 24 priority citations are real, verifiable papers with metadata matching the draft's claims (modulo the minor refinements above).

---

## Needs deeper verification

**None within the priority list.** No citation required a third search attempt.

---

## Audit summary

- 24/24 citations verified as real papers. 0 hallucinations.
- 4 minor wording refinements suggested (none changes citation correctness).
- 6 optional venue upgrades suggested (peer-reviewed venue exists; cited as arXiv only).
- "Spyra" surname — flagged as suspicious in priority list — IS real (Polish, single-author).
- "Asperti" — flagged Italian — IS Italian (Bologna). Verified.
- arxiv 2602.12009 (Feb 2026 paper flagged as plausible-future-dated) — verified, ICASSP 2026 to appear.
- Future-dated papers 2510.04595, 2511.21181, 2601.00853, 2602.12009, 2601.22076 — all verified real.

Time used: ~50 min (under 90 min budget).
