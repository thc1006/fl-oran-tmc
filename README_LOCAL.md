# fl_oran — Local PyTorch re-implementation

Local, reproducible re-implementation of the three Colosseum-ORAN federated
learning notebooks (`notebooks/*.ipynb`), tuned to squeeze maximum performance
out of a single RTX 4080 workstation.

| Variant | Notebook | Model | Target(s) | Notes |
|---------|----------|-------|-----------|-------|
| **v106** | `colosseum_oran_federated_slicing_v1_0_6.ipynb` | MLP 13 → 64 → 32 → 1 (Dropout) | `allocation_efficiency` | Classic baseline |
| **v107_1** | `colosseum_oran_federated_slicing_v1_0_7-1.ipynb` | LSTM(64) → LSTM(32) → FC(64) → head | 3 regression + 1 classification (combined head) | Temporal sliding window (seq_len=5); trend features derived on-the-fly |
| **v107_2** | `colosseum_oran_federated_slicing_v1_0_7-2.ipynb` | MLP 13 → 128(BN) → 64(BN) → 32 → 1 | `allocation_efficiency` | Enhanced MLP + DP (Gaussian mechanism) |

## Differences captured from the notebooks

- **v1.0.6**: standard single-output regression. Small MLP, per-client
  StandardScaler with weighted-average global scaler, stratified sampling on
  the target, custom DP optimizer wrapper (disabled by default in the notebook).
- **v1.0.7-1**: **multi-output temporal model**. Builds trend features
  (`req_prbs_last3`, `req_prbs_change_rate`, `req_prbs_volatility`,
  `is_peak_hour`, `is_weekend`) and many-to-one sliding windows (seq_len=5).
  Uses LSTM×2 + FC + combined head over 3 regression + 1 classification (`sla_violation`). CuPy-based scalers in the original; we use vectorised NumPy on 32 CPU cores which is already very fast.
- **v1.0.7-2**: **enhanced MLP + advanced DP**. Deeper MLP with BatchNorm,
  domain-knowledge data-quality checks, adaptive clipping DP optimizer, and
  an advanced per-client privacy budget manager.

All three originally rely on TensorFlow 2.14.1 + TensorFlow Federated 0.86.0,
which have a brittle dependency chain and are much slower than a hand-rolled
FedAvg on modern PyTorch. This repo drops TFF and implements:

- **Pure PyTorch FedAvg**: server-side weighted averaging of client state_dicts (`src/fl_oran/federated/aggregation.py`).
- **Gaussian DP on server updates** with RDP accounting (`federated/dp.py`).
- **AMP (bf16 by default)** + `torch.compile` (reduce-overhead) + TF32 matmul for the 4080.
- **DataLoader with `num_workers=8`, `pin_memory`, `prefetch_factor=4`** to saturate PCIe and feed the GPU.

## Repo layout

```
.
├── pyproject.toml              # PEP 621 project
├── data/coloran_processed_features.parquet   # symlink to the raw parquet
├── notebooks/                  # original .ipynb files (untouched)
├── src/fl_oran/                # main package
│   ├── cli.py                  # CLI entry point
│   ├── config.py               # dataclass configs
│   ├── logging_utils.py        # Rich-based logger
│   ├── data/                   # loader, scalers, sequences, quality, trend features
│   ├── models/                 # MLPv106, MLPv107_2, LSTMMultiOutput
│   ├── federated/              # FedAvg, DP (Gaussian), PrivacyAccountant, client trainer
│   ├── training/               # FederatedTrainer — orchestration
│   ├── evaluation/             # regression & multi-output metrics
│   └── utils/                  # seeding, GPU/AMP helpers
├── experiments/
│   ├── configs/                # v106.yaml, v107_1.yaml, v107_2.yaml
│   └── run_v10{6,7_1,7_2}.py   # thin wrappers
├── scripts/
│   ├── setup.sh                # bootstrap a .venv with uv
│   ├── smoke_test.sh           # 3-round run on 1% data for each variant
│   └── run_all.sh              # full runs
├── tests/                      # pytest + pytest-cov; TDD-style coverage
└── artifacts/                  # models/, logs/, plots/ (gitignored)
```

## Quick start

```bash
# 1. Clone and bootstrap the venv (one-time):
./scripts/setup.sh

# 2. Activate:
source .venv/bin/activate

# 3. Run the whole test suite:
pytest -q

# 4. Smoke test all three variants on 1% data (<5 min):
./scripts/smoke_test.sh

# 5. Full runs (backed by YAML configs):
python experiments/run_v106.py
python experiments/run_v107_1.py
python experiments/run_v107_2.py

# Equivalent CLI:
python -m fl_oran --variant v106    --num-rounds 30 --samples-per-client 200000
python -m fl_oran --variant v107_1  --num-rounds 30 --samples-per-client 100000 --seq-len 5
python -m fl_oran --variant v107_2  --num-rounds 30 --samples-per-client 200000 --dp
```

## Performance tuning for the RTX 4080

Defaults in `TrainingConfig`:

- `mixed_precision="bf16"` — bf16 autocast; Ada (sm_89) has native bf16 tensor cores. Faster than fp32, no loss scaling, no NaNs.
- `compile_model=True` — `torch.compile(mode="reduce-overhead")` reduces kernel launch overhead for small MLPs.
- `batch_size=512` for MLPs, `256` for the LSTM.
- `num_workers=8, prefetch_factor=4, persistent_workers=True` — saturates the PCIe link; 32 CPU cores feed the GPU easily.
- `torch.set_float32_matmul_precision("high")` — enables TF32 matmul for any remaining fp32 paths.

If VRAM runs out:
- Lower `--batch-size`, or
- Lower `--samples-per-client`.

## Data

Place the ColO-RAN processed parquet at `data/coloran_processed_features.parquet`
(a symlink in this repo already points at the project root copy). 35,512,393 rows × 16 numeric features + 2 categorical, 7 base stations (clients), no nulls.

## Logging

Every run writes a timestamped log under `artifacts/logs/<run_name>_<ts>.log`
plus a training history CSV at `artifacts/logs/<run_name>_history.csv`. The
final/best model weights go to `artifacts/models/<run_name>_best.pt` and the
effective config is dumped to `artifacts/logs/<run_name>_config.json`.

`FL_ORAN_LOG_LEVEL=DEBUG` and `FL_ORAN_ARTIFACTS=/other/dir` are honoured.

## Tests & coverage

```bash
pytest -q                        # fast tests
pytest -m "not slow" -q          # skip slow tests
pytest --cov=src/fl_oran         # with coverage (html in artifacts/coverage/)
```

All core modules are covered (data loading, scalers, sequences, models, FedAvg
aggregation, DP mechanism, metrics, client trainer, and a smoke-test for the
full FL loop on synthetic data).

## License

Released under AGPL-3.0 — see `LICENSE`. ColO-RAN dataset credit:
M. Polese *et al.*, “ColO-RAN: Developing Machine Learning-based xApps for Open RAN Closed-loop Control,” IEEE TMC, 2022.
