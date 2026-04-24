"""v5 algorithm-sweep orchestrator.

One call to ``run_v5_sweep(cfg)`` executes a single (algorithm, alpha, seed)
cell of the paper matrix:

1. Load the unified ColO-RAN parquet, build the classification target, OOD
   split by ``training_config``.
2. Partition the train split across ``n_clients`` using Dirichlet over
   ``slice_id``.
3. Fit a federated continuous-feature scaler (sufficient-stats
   aggregation) and scale all splits.
4. Instantiate the FL algorithm via the registry. MOON gets
   ``forecaster_encode_fn`` injected automatically; other algorithms use
   their own ``__init__`` kwargs from ``cfg.algo_kwargs``.
5. Run ``num_rounds`` of federated training. Each round: sample a client
   subset, build a fresh local model loaded with the global state, call
   ``algo.client_update`` per client, then ``algo.server_aggregate``.
6. Track best val-AUC; at the end, restore best weights and evaluate on
   the held-out test split.
7. Emit ``artifacts/v5_sweep/<name>/{summary.json, history.csv, best.pt}``.

The function is CLI-agnostic — the CLI wrapper lives in
``experiments/run_v5_algorithm_sweep.py``. This file holds the reusable
library code so tests can drive it directly.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import nn

from ..data_v2.encoders import (
    FeatureSchema,
    apply_continuous_scaler,
    federated_fit_scaler,
)
from ..data_v2.partition import partition_clients
from ..data_v2.sequences import build_run_sequences
from ..data_v2.split import ood_split_by_tr
from ..data_v2.targets_v2 import add_classification_target
from ..federated.algorithms import get_algorithm
from ..logging_utils import get_logger
from ..models.forecaster_v2 import ForecasterV2
from ..utils import autocast_dtype, log_cuda_info, pick_device, seed_everything
from .centralized_v3 import (
    V3_CATEGORICAL,
    V3_CAT_SIZES,
    V3_CONTINUOUS,
    _batched_predict,
    _metrics,
)

log = get_logger(__name__)


# --------------------------------------------------------------------------
# MOON representation extraction for ForecasterV2.
#
# ForecasterV2's pipeline is:
#   embed(cats) + x_cont -> LSTM1 -> LSTM2 -> last = h[:, -1, :]
#                       -> fc(dropout(last)) -> relu -> head -> logits
#
# We use the post-fc post-ReLU 64-dim activation as the representation.
# Dropout is bypassed so z is deterministic regardless of the host model's
# train/eval mode — this keeps the contrastive similarity consistent
# between the live (train-mode) local model and the frozen (eval-mode)
# global / prev snapshots.
# --------------------------------------------------------------------------


def forecaster_encode_fn(
    model: ForecasterV2,
    x_cat: torch.Tensor,
    x_cont: torch.Tensor,
) -> torch.Tensor:
    """Return the penultimate representation for a ForecasterV2 instance.

    Shape: ``(B, fc_hidden)`` (64 by default). Gradient flows through
    ``embeddings``, ``lstm1``, ``lstm2``, and ``fc`` but not through the
    final classification ``head``.
    """
    cats = [model.embeddings[col](x_cat[..., i])
            for i, col in enumerate(model.schema.categorical)]
    x = torch.cat(cats + [x_cont], dim=-1) if cats else x_cont
    h, _ = model.lstm1(x)
    h, _ = model.lstm2(h)
    last = h[:, -1, :]
    return model.relu(model.fc(last))


# --------------------------------------------------------------------------
# Configuration.
# --------------------------------------------------------------------------


@dataclass
class V5Config:
    # Run identification (``name`` auto-generated in __post_init__ if empty).
    name: str = ""

    # Algorithm selection.
    algorithm: str = "fedavg"  # key into REGISTRY
    algo_kwargs: dict[str, Any] = field(default_factory=dict)

    # Partition.
    partition_mode: str = "dirichlet"
    alpha: float = 0.5
    n_clients: int = 5

    # Training.
    num_rounds: int = 20
    clients_per_round: int = 5
    max_steps_per_round: int = 50
    batch_size: int = 64
    lr: float = 5e-4
    lr_warmup_rounds: int = 3
    grad_clip: float = 1.0

    # Data.
    unified_parquet: Path = field(
        default_factory=lambda: Path("data/coloran_raw_unified.parquet")
    )
    sample_ratio: float = 1.0
    threshold: float = 0.10
    seq_len: int = 5
    train_tr: list[int] = field(default_factory=lambda: list(range(22)))
    val_tr: list[int] = field(default_factory=lambda: [22, 23, 24])
    test_tr: list[int] = field(default_factory=lambda: [25, 26, 27])

    # System.
    seed: int = 42
    device: str = "cuda"
    mixed_precision: str = "bf16"
    compile_model: str | None = None
    # Which split drives pos_weight for BCEWithLogitsLoss. v4 baseline used
    # "test" (leaks test-set prior into training loss). v5 default is
    # "train" — scientifically correct. Switch back to "test" to match v4
    # numbers exactly.
    pos_weight_split: str = "train"
    # If True, force deterministic cuDNN algorithms at the cost of ~5-15%
    # LSTM throughput. Recommended for reproducibility-critical runs.
    cudnn_deterministic: bool = True
    output_dir: Path = field(
        default_factory=lambda: Path("artifacts/v5_sweep")
    )

    def __post_init__(self) -> None:
        if not self.name:
            alpha_tag = f"{self.alpha:.2f}".replace(".", "p")
            self.name = f"v5_{self.algorithm}_a{alpha_tag}_s{self.seed}"
        self.unified_parquet = Path(self.unified_parquet)
        self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unified_parquet"] = str(self.unified_parquet)
        d["output_dir"] = str(self.output_dir)
        return d


# --------------------------------------------------------------------------
# Orchestrator.
# --------------------------------------------------------------------------


def _partition(df: pd.DataFrame, cfg: V5Config) -> dict[int, pd.DataFrame]:
    """Dispatch to ``partition_clients`` with the v5 config's mode choice."""
    if cfg.partition_mode == "dirichlet":
        return partition_clients(
            df, mode="dirichlet",
            alpha=cfg.alpha, n_clients=cfg.n_clients, seed=cfg.seed,
        )
    if cfg.partition_mode == "iid":
        return partition_clients(df, mode="iid")
    raise ValueError(
        f"unsupported partition_mode for v5: {cfg.partition_mode!r} "
        "(use 'dirichlet' or 'iid')"
    )


@dataclass
class PreparedData:
    """Everything the training loop needs, decoupled from data ingestion.

    Building this takes ~55s for the full ColO-RAN parquet (18M rows) but
    is identical across the 6 algorithms of a given (seed, alpha) cell.
    The matrix driver (``experiments/run_v5_sweep_matrix.py``) builds it
    once and feeds ``_run_training`` repeatedly — 6x fewer parquet reads
    + sequence builds per sweep cell.
    """
    schema: FeatureSchema
    client_cpu: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    val_cat: torch.Tensor
    val_cont: torch.Tensor
    val_y: torch.Tensor
    test_cat: torch.Tensor
    test_cont: torch.Tensor
    test_y: torch.Tensor
    # pos_weight computed from the test split (matches fl_v3 convention;
    # see docstring of run_v5_sweep for the leakage caveat).
    pos_weight: torch.Tensor


def prepare_v5_data(
    cfg: V5Config,
    device: torch.device,
    *,
    n_jobs_sequences: int = 5,
) -> PreparedData:
    """Load + target + split + Dirichlet + sequences + scaler, once.

    Per-client ``build_run_sequences`` runs in a joblib thread pool
    (``n_jobs_sequences`` workers, threading backend so pandas/NumPy share
    the DataFrame via memory). Threading is correct because groupby +
    sliding_window_view are GIL-releasing NumPy ops.

    The returned ``PreparedData`` is CPU-resident (tensors are pinned when
    ``device.type == "cuda"`` for overlap with H2D copies). The training
    loop moves per-client tensors onto the GPU on demand.
    """
    if not cfg.unified_parquet.exists():
        raise FileNotFoundError(cfg.unified_parquet)
    t0 = time.time()
    df = pd.read_parquet(cfg.unified_parquet)
    if cfg.sample_ratio < 1.0:
        df = (df.sample(frac=cfg.sample_ratio, random_state=cfg.seed)
                .sort_index().reset_index(drop=True))
    df = add_classification_target(
        df, column="ul_bler", threshold=cfg.threshold, target_name="y_sla_next"
    )
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    feat_cols = schema.categorical + schema.continuous
    split = ood_split_by_tr(df, cfg.train_tr, cfg.val_tr, cfg.test_tr)

    # Partition train across clients via Dirichlet over slice_id.
    client_dfs = _partition(split.train, cfg)
    client_items = list(client_dfs.items())

    # Per-client sequence build in parallel. Threading backend reuses the
    # process memory (no pickling overhead) — pandas groupby + NumPy
    # sliding_window_view both release the GIL.
    n_workers = min(n_jobs_sequences, len(client_items)) or 1
    per_client_results = Parallel(n_jobs=n_workers, backend="threading")(
        delayed(build_run_sequences)(
            d, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len,
        )
        for _, d in client_items
    )
    client_shards: dict[int, tuple[np.ndarray, np.ndarray]] = {
        cid: (X, Y)
        for (cid, _), (X, Y) in zip(client_items, per_client_results)
        if len(X) > 0
    }
    if not client_shards:
        raise RuntimeError(
            f"no non-empty clients after partition+sequence build "
            f"(alpha={cfg.alpha}, n_clients={cfg.n_clients})"
        )
    log.info("v5 prep: %d non-empty clients  rows/client=%s",
             len(client_shards),
             {c: len(x) for c, (x, _) in client_shards.items()})

    X_va, Y_va = build_run_sequences(
        split.val, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len
    )
    X_te, Y_te = build_run_sequences(
        split.test, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len
    )

    scaler = federated_fit_scaler(
        {cid: X for cid, (X, _) in client_shards.items()}, schema
    )

    def _to_tensors(X: np.ndarray, Y: np.ndarray):
        cat, cont = apply_continuous_scaler(X, schema, scaler)
        return (torch.from_numpy(cat), torch.from_numpy(cont), torch.from_numpy(Y))

    def _maybe_pin(t: torch.Tensor) -> torch.Tensor:
        return t.pin_memory() if device.type == "cuda" else t

    va_cat, va_cont, va_y = _to_tensors(X_va, Y_va)
    te_cat, te_cont, te_y = _to_tensors(X_te, Y_te)
    va_cat = _maybe_pin(va_cat); va_cont = _maybe_pin(va_cont)
    te_cat = _maybe_pin(te_cat); te_cont = _maybe_pin(te_cont)
    client_cpu: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for cid, (X, Y) in client_shards.items():
        cat, cont = apply_continuous_scaler(X, schema, scaler)
        client_cpu[cid] = (
            _maybe_pin(torch.from_numpy(cat)),
            _maybe_pin(torch.from_numpy(cont)),
            _maybe_pin(torch.from_numpy(Y)),
        )

    # pos_weight: compute from the requested split.
    if cfg.pos_weight_split == "train":
        # Union of client Y tensors (== train positive rate).
        train_pos = sum(int((Y > 0.5).sum()) for _, (_, Y) in client_shards.items())
        train_n = sum(int(len(Y)) for _, (_, Y) in client_shards.items())
        pos_rate = max(train_pos / max(train_n, 1), 1e-6)
    elif cfg.pos_weight_split == "test":
        pos_rate = max(float(Y_te.mean()), 1e-6)
    else:
        raise ValueError(
            f"pos_weight_split must be 'train' or 'test', got {cfg.pos_weight_split!r}"
        )
    pos_weight = torch.tensor(
        [max((1 - pos_rate) / pos_rate, 1.0)], dtype=torch.float32,
    )

    log.info("v5 prep complete: %.1fs  train_clients=%d  val=%d  test=%d",
             time.time() - t0, len(client_cpu), len(va_y), len(te_y))
    return PreparedData(
        schema=schema,
        client_cpu=client_cpu,
        val_cat=va_cat, val_cont=va_cont, val_y=va_y,
        test_cat=te_cat, test_cont=te_cont, test_y=te_y,
        pos_weight=pos_weight,
    )


def _build_algorithm(cfg: V5Config, amp_enabled: bool, amp_dtype):
    """Instantiate the FL algorithm from the registry, injecting encode_fn for MOON."""
    algo_cls = get_algorithm(cfg.algorithm)
    kwargs: dict[str, Any] = {
        "max_steps": cfg.max_steps_per_round,
        "batch_size": cfg.batch_size,
        "grad_clip": cfg.grad_clip,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
    }
    kwargs.update(cfg.algo_kwargs)
    if cfg.algorithm == "moon":
        kwargs.setdefault("encode_fn", forecaster_encode_fn)
    return algo_cls(**kwargs)


def setup_torch_perf(device: torch.device, deterministic: bool = False) -> None:
    """Idempotent workstation-level perf switches. Safe to call repeatedly.

    ``deterministic=True`` forces cuDNN to pick algorithms that produce
    identical output across runs (required for full bit-level
    reproducibility); incompatible with ``cudnn.benchmark`` so the latter
    is disabled in that case.
    """
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _run_training(
    cfg: V5Config,
    data: PreparedData,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype,
) -> dict:
    """Training loop + test eval + artifact emission, given ``PreparedData``.

    Used by both ``run_v5_sweep`` (single-cell CLI) and the matrix driver
    that shares ``data`` across a row of the algorithm matrix.
    """
    schema = data.schema

    def build_model() -> ForecasterV2:
        m = ForecasterV2(
            schema=schema, task="classification", seq_len=cfg.seq_len,
        ).to(device)
        if cfg.compile_model and device.type == "cuda":
            m = torch.compile(m, mode=cfg.compile_model)
        return m

    global_model = build_model()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=data.pos_weight.to(device))
    algo = _build_algorithm(cfg, amp_enabled, amp_dtype)

    cids = sorted(data.client_cpu.keys())
    rng = np.random.default_rng(cfg.seed)
    history: list[dict] = []
    # Use -inf so that ANY first-round val AUC (even exactly 0.0 / NaN-fallback)
    # promotes to best_state. Prevents "last round == best" when val is
    # degenerate (single-class; AUC absent).
    best_val_auc: float = float("-inf")
    best_state: dict | None = None

    for r in range(1, cfg.num_rounds + 1):
        t0 = time.time()
        k = min(cfg.clients_per_round, len(cids))
        selected = rng.choice(cids, size=k, replace=False).tolist()
        # Keep global_state on-device so clients load GPU-GPU (no D2H/H2D bounce).
        global_state = {k_: v.detach().clone()
                        for k_, v in global_model.state_dict().items()}
        lr_this = cfg.lr * min(1.0, r / max(cfg.lr_warmup_rounds, 1))

        updates = []
        for cid in selected:
            local_model = build_model()
            local_model.load_state_dict(global_state, strict=True)
            update = algo.client_update(
                client_id=int(cid),
                local_model=local_model,
                client_tensors=data.client_cpu[cid],
                loss_fn=loss_fn,
                current_lr=lr_this,
                device=device,
                round_idx=r,
            )
            updates.append(update)
            del local_model
            # NOTE: torch.cuda.empty_cache() removed from the per-client loop —
            # it forces a CUDA synchronisation on every call and costs
            # ~50-100ms per invocation with no memory benefit at our scale.

        new_state = algo.server_aggregate(
            global_state=global_state, updates=updates,
        )
        global_model.load_state_dict(new_state)

        val_logits = _batched_predict(global_model, data.val_cat, data.val_cont, device)
        val_m = _metrics(data.val_y[:, 0].numpy().astype(int), val_logits[:, 0])
        train_l = float(np.mean([u.train_loss for u in updates]))
        dt = time.time() - t0
        history.append({
            "round": r,
            "train_loss": train_l,
            "val_auc": val_m.get("auc", 0.0),
            "val_acc": val_m["accuracy"],
            "val_f1": val_m["f1"],
            "lr": lr_this,
            "duration_s": dt,
        })
        log.info(
            "%s r%d/%d  train=%.4f  val_auc=%.4f  val_acc=%.4f  dt=%.1fs",
            cfg.name, r, cfg.num_rounds, train_l,
            val_m.get("auc", 0.0), val_m["accuracy"], dt,
        )
        if val_m.get("auc", 0.0) > best_val_auc:
            best_val_auc = val_m["auc"]
            best_state = {k_: v.detach().cpu()
                          for k_, v in global_model.state_dict().items()}

    # Test evaluation on the best val-AUC checkpoint.
    if best_state is not None:
        global_model.load_state_dict(best_state)
    test_logits = _batched_predict(global_model, data.test_cat, data.test_cont, device)
    test_m = _metrics(data.test_y[:, 0].numpy().astype(int), test_logits[:, 0])

    # --- Emit artifacts ---
    import platform
    env_meta = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": (torch.backends.cudnn.version()
                  if torch.cuda.is_available() else None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
    }
    result = {
        "config": cfg.to_dict(),
        "env": env_meta,
        "history": history,
        "best_val_auc": best_val_auc,
        "test": test_m,
    }
    run_dir = cfg.output_dir / cfg.name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "summary.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    pd.DataFrame(history).to_csv(
        run_dir / "logs" / "history.csv", index=False
    )
    if best_state is not None:
        torch.save(best_state, run_dir / "models" / "best.pt")

    log.info(
        "%s done: best_val_auc=%.4f  test_auc=%.4f  test_acc=%.4f  test_f1=%.4f",
        cfg.name, best_val_auc,
        test_m.get("auc", 0.0), test_m["accuracy"], test_m["f1"],
    )
    return result


def run_v5_sweep(cfg: V5Config) -> dict:
    """Run one sweep cell (1 algorithm, 1 alpha, 1 seed).

    Convenience wrapper that loads data and runs training in a single call.
    For multi-algorithm sweeps use ``experiments/run_v5_sweep_matrix.py``
    which shares ``prepare_v5_data`` across a row of the matrix (massive
    speedup at full-sweep scale).
    """
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    log_cuda_info(device)
    setup_torch_perf(device, deterministic=cfg.cudnn_deterministic)
    amp_enabled, amp_dtype = autocast_dtype(cfg.mixed_precision)

    data = prepare_v5_data(cfg, device)
    return _run_training(cfg, data, device, amp_enabled, amp_dtype)
