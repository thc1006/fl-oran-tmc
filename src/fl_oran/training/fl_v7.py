"""fl_v7: arch-agnostic FL trainer for Stage 2 (3 archs × N algos × N partitions).

ADR-001 D-22 Phase 1.5b implementation. Parallel to ``fl_v5.py`` (which is
hard-bound to ``ForecasterV2`` / LSTM); fl_v5 stays unchanged per D-9.
fl_v7 dispatches model construction via ``_build_model`` against the
single-source-of-truth ``ARCH_REGISTRY`` from
``experiments/run_v6_arch_sweep.py`` (importlib pattern, no sys.path
mutation, mirrors ``scripts/_v6_cell_metadata.py``).

**Performance inheritance from M5** (see ADR D-22 perf checklist):

* ``mixed_precision="bf16"`` — RTX 4080 Ada native
* ``setup_torch_perf`` — TF32 matmul + cudnn flags
* fused Adam on CUDA (driven by algorithms' optimiser construction)
* ``compile_model: str | None`` arch-conditional default via
  :func:`_select_compile_mode` — "reduce-overhead" (CUDA Graphs) for
  dense archs (lstm, mamba), ``None`` for spiking archs to avoid
  graph-break under nested Python for-loops in
  ``SpikingSSMBlock._scan_emit_spikes``
* ``cudnn_deterministic=True`` — D-15 reproducibility mandate
* ``non_blocking=True`` for all CPU→GPU tensor transfers
* ``federated_fit_scaler(n_jobs=...)`` — per-client CPU parallelism
  (joblib threading, GIL released by NumPy ops); inherited via D-3 reuse

**MOON treatment** (D-16 open question): ``_select_algorithm`` raises
``NotImplementedError`` for ``algorithm="moon"`` regardless of arch.
The encode_fn for spiking/mamba is paper-level design deferred to
Phase 2 polish.

**Output layout** (ADR D-7): ``cfg.output_dir / cfg.name /
{summary.json, history.csv, best.pt}`` flat — no ``/logs`` or
``/models`` subdirectories (deviates from fl_v5's nested layout to
match D-7's spec for v7).
"""
from __future__ import annotations

import importlib.util
import json
import platform
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
from ..utils import autocast_dtype, log_cuda_info, pick_device, seed_everything
from .centralized_v3 import (
    V3_CATEGORICAL,
    V3_CAT_SIZES,
    V3_CONTINUOUS,
    _batched_predict,
    _metrics,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Arch registry — single source of truth via importlib (no sys.path mutation).
# ---------------------------------------------------------------------------

# repo_root/src/fl_oran/training/fl_v7.py → repo_root/experiments/run_v6_arch_sweep.py
_RUNNER_PATH = (
    Path(__file__).resolve().parents[3] / "experiments" / "run_v6_arch_sweep.py"
)
_RUNNER_CACHE: dict | None = None


def _arch_registry() -> dict:
    """Load and cache ARCH_REGISTRY from run_v6_arch_sweep.py.

    Avoids drift between the centralized v6 sweep and fl_v7 — both walk
    the same registry. Cached at module scope so a 1050-cell sweep only
    pays the import cost once.
    """
    global _RUNNER_CACHE
    if _RUNNER_CACHE is None:
        spec = importlib.util.spec_from_file_location("_v6_runner", _RUNNER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load runner module from {_RUNNER_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RUNNER_CACHE = mod.ARCH_REGISTRY
    return _RUNNER_CACHE


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

@dataclass
class V7Config:
    """Stage 2 FL × architecture sweep config.

    Mirrors V5Config conventions plus the new ``arch`` axis. Per ADR D-22
    perf checklist, all 4 perf-related fields inherit M5 defaults so
    fl_v7 sweep wallclock matches M5 baseline.
    """
    # Run identification (auto-generated in __post_init__ if empty).
    name: str = ""

    # NEW: arch axis (the entire reason fl_v7 exists vs fl_v5).
    arch: str = "lstm"
    arch_kwargs: dict[str, Any] = field(default_factory=dict)

    # Algorithm.
    algorithm: str = "fedavg"
    algo_kwargs: dict[str, Any] = field(default_factory=dict)

    # Partition.
    partition_mode: str = "dirichlet"
    alpha: float = 0.5
    n_clients: int = 7  # ADR §5: matches ColO-RAN gNB count

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

    # System (perf inheritance from M5 — see module docstring).
    seed: int = 42
    device: str = "cuda"
    mixed_precision: str = "bf16"
    compile_model: str | None = None  # arch-conditional default in _select_compile_mode
    pos_weight_split: str = "train"  # D-12 audit fix; "test" was leakage
    cudnn_deterministic: bool = True  # D-15 mandate
    output_dir: Path = field(
        default_factory=lambda: Path("artifacts/v7_fl_arch_sweep")
    )

    def __post_init__(self) -> None:
        if not self.name:
            # Partition + n_clients aware auto-name: EVERY dimension
            # that affects the produced data must be in the name to
            # prevent silent cell-directory overwrite when matrix
            # sweeps cross multiple values. The legacy v7 format
            # ``a<alpha>`` alone collided when running IID + Dirichlet
            # at the same default alpha=0.5 (alpha is meaningless for
            # IID but lived in cfg). For future n_clients ablation
            # sweeps it would also collide on n=5 vs n=7 in dirichlet.
            #
            # Format per partition_mode:
            #   iid:        v7_<arch>_<algo>_iid_n<N>_s<seed>
            #   dirichlet:  v7_<arch>_<algo>_dirichlet_a<alpha>_n<N>_s<seed>
            #   other:      v7_<arch>_<algo>_<mode>_a<alpha>_n<N>_s<seed>
            if self.partition_mode == "iid":
                part_tag = f"iid_n{self.n_clients}"
            elif self.partition_mode == "dirichlet":
                alpha_tag = f"{self.alpha:.2f}".replace(".", "p")
                part_tag = f"dirichlet_a{alpha_tag}_n{self.n_clients}"
            else:
                # Fallback for future modes (e.g. noniid_slice): keep
                # both alpha + n_clients for safety. Update this
                # branch when adding a mode where alpha is meaningless.
                alpha_tag = f"{self.alpha:.2f}".replace(".", "p")
                part_tag = (
                    f"{self.partition_mode}_a{alpha_tag}_n{self.n_clients}"
                )
            self.name = (
                f"v7_{self.arch}_{self.algorithm}_{part_tag}_s{self.seed}"
            )
        self.unified_parquet = Path(self.unified_parquet)
        self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unified_parquet"] = str(self.unified_parquet)
        d["output_dir"] = str(self.output_dir)
        return d


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _build_model(cfg: V7Config, schema: FeatureSchema) -> nn.Module:
    """Arch-agnostic model factory via run_v6_arch_sweep ARCH_REGISTRY.

    Returns model on CPU; caller moves to device. Layer order:

      1. Registry kwargs for ``cfg.arch`` (e.g. ``backbone_d_model=56,
         backbone_expand=2`` for spiking_expand2)
      2. ``cfg.arch_kwargs`` overrides (for HPO sweeps)
      3. Standard kwargs from V7Config (schema, task, seq_len)

    Raises ValueError with the offending arch name in the message if
    ``cfg.arch`` is not in the registry.
    """
    registry = _arch_registry()
    if cfg.arch not in registry:
        raise ValueError(
            f"unknown arch={cfg.arch!r}; "
            f"known archs are {sorted(registry.keys())}"
        )
    arch_cfg = registry[cfg.arch]
    ctor = arch_cfg["ctor"]
    kwargs = dict(arch_cfg.get("kwargs", {}))
    kwargs.update(cfg.arch_kwargs)
    return ctor(
        schema=schema, task="classification", seq_len=cfg.seq_len, **kwargs,
    )


# Per-algorithm required kwargs (post-fl_v7-auto-fill). ``run_v7_sweep``
# already injects max_steps/batch_size/grad_clip/amp_enabled/amp_dtype
# (see _run_training_v7), so this table only lists user-supplied
# algorithm hyperparameters that the spec MUST provide. The
# spec loader (scripts/_v7_spec_loader.py) consults this table at
# load time to refuse a spec that lacks a required kwarg — that is
# how Phase 2's "all 18 fedprox cells crashed at training time" bug
# class becomes undeployable.
#
# MOON is intentionally absent: its encode_fn requirement is a deferred
# paper-level question per ADR D-22 / D-16, and ``_select_algorithm``
# raises NotImplementedError before reaching this table.
_ALGO_REQUIRED_KWARGS: dict[str, set[str]] = {
    "fedavg":   set(),
    "fedprox":  {"mu"},
    "fedadam":  {"server_lr"},
    "scaffold": set(),
    "feddyn":   {"alpha"},  # FedDyn regularization; NOT V7Config.alpha (Dirichlet)
}


def _select_algorithm(cfg: V7Config):
    """Return the algorithm class, fail-fast on MOON × any-arch.

    Phase 1.5 (per D-22) defers MOON entirely because ``encode_fn`` for
    spiking/mamba is a paper-level open question (D-16). Implementing
    only MOON × LSTM by importing fl_v5's ``forecaster_encode_fn``
    creates undesirable v5↔v7 coupling. Cleaner to defer all MOON to
    Phase 2 polish.
    """
    if cfg.algorithm == "moon":
        raise NotImplementedError(
            "MOON encode_fn for arch-agnostic FL is an open paper-level "
            "design question (ADR D-16); deferred to Phase 2 polish. "
            f"For Phase 1.5 use a non-MOON algorithm with arch={cfg.arch!r}."
        )
    return get_algorithm(cfg.algorithm)


def _select_compile_mode(cfg: V7Config) -> str | None:
    """Arch-conditional ``torch.compile`` mode default.

    Explicit ``cfg.compile_model`` override always wins. Otherwise:

    * dense archs (lstm, mamba, mamba_expand2) → ``"reduce-overhead"``
      (CUDA Graphs) — significant speedup on RTX 4080 for static-shape
      forward.
    * spiking archs (anything starting with ``"spiking"``) → ``None``.
      ``SpikingForecaster._scan_emit_spikes`` has nested Python
      for-loops + stateful LIF ``mem`` that graph-break under
      reduce-overhead — using it would silently regress wallclock
      OR raise a runtime error.
    """
    if cfg.compile_model is not None:
        return cfg.compile_model
    return None if cfg.arch.startswith("spiking") else "reduce-overhead"


def setup_torch_perf(device: torch.device, deterministic: bool = True) -> None:
    """Idempotent workstation perf switches; mirror ``fl_v5.setup_torch_perf``.

    TF32 matmul precision="high" gives ~2-3× speedup on Ada Lovelace
    (sm_89) for fp32 matmul ops. ``cudnn_deterministic=True`` costs
    ~5-15% LSTM throughput but is mandatory per ADR D-15.

    CPU thread tuning (M5-baseline alignment): clamp
    ``torch.set_num_threads`` to 8 to prevent contention with joblib
    threading layer used by ``federated_fit_scaler`` (n_jobs=n_clients
    up to 7) and per-client ``build_run_sequences``. Without this,
    16-core workstations oversubscribe → ~2× slowdown on CPU prep
    stages.
    """
    # CPU thread tuning applies to both CPU and CUDA paths (matmul on
    # CUDA, prep stages on CPU). Idempotent (safe to call repeatedly).
    try:
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    except Exception:
        # Some build configurations don't expose set_num_threads; ignore.
        pass

    if device.type != "cuda":
        return
    # "medium" allows BF16 reductions for fp32 matmul on Ada (sm_89) —
    # measurably faster than "high" (TF32 reductions only) at the cost
    # of ~3-4 ULP precision in matmul output. With our AMP autocast
    # already operating in bf16 the precision impact is moot.
    torch.set_float32_matmul_precision("medium")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _partition(df: pd.DataFrame, cfg: V7Config) -> dict[int, pd.DataFrame]:
    """Dispatch to ``partition_clients`` with the cfg's mode choice.

    Per partition.py contract: ``mode="iid"`` partitions by ``bs_id``
    (=7 ColO-RAN gNBs) and **ignores** ``cfg.n_clients``. ``mode=
    "dirichlet"`` uses ``cfg.n_clients`` × per-slice Dirichlet over
    ``cfg.alpha`` — canonical Stage 2 partition per ADR D-17.
    """
    if cfg.partition_mode == "dirichlet":
        return partition_clients(
            df, mode="dirichlet",
            alpha=cfg.alpha, n_clients=cfg.n_clients, seed=cfg.seed,
        )
    if cfg.partition_mode == "iid":
        return partition_clients(df, mode="iid")
    raise ValueError(
        f"unsupported partition_mode for v7: {cfg.partition_mode!r} "
        "(use 'dirichlet' or 'iid')"
    )


# ---------------------------------------------------------------------------
# Main entry: run_v7_sweep.
# ---------------------------------------------------------------------------

def run_v7_sweep(cfg: V7Config) -> dict:
    """Run one sweep cell (1 arch × 1 algorithm × 1 alpha × 1 seed).

    Mirrors ``fl_v5.run_v5_sweep`` but uses ``_build_model`` for arch
    dispatch and ``_select_compile_mode`` for arch-conditional torch.compile.

    Output layout (ADR D-7): ``cfg.output_dir / cfg.name /
    {summary.json, history.csv, best.pt}``.
    """
    # All RNG sources seeded from cfg.seed (D-15).
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    log_cuda_info(device)
    setup_torch_perf(device, deterministic=cfg.cudnn_deterministic)
    amp_enabled, amp_dtype = autocast_dtype(cfg.mixed_precision)

    # Fail-fast on MOON before parquet load (helper raises NotImplementedError).
    algo_cls = _select_algorithm(cfg)

    # ---- Data preparation ----
    # Phase-level timing instrumentation: each phase recorded separately
    # so post-sweep analysis can see where time goes (per ADR D-22 perf
    # checklist). Total reported at end + each phase emits a single INFO
    # log line.
    phase_timings: dict[str, float] = {}

    if not cfg.unified_parquet.exists():
        raise FileNotFoundError(cfg.unified_parquet)

    t0 = time.time()
    t_phase = time.time()
    df = pd.read_parquet(cfg.unified_parquet)
    if cfg.sample_ratio < 1.0:
        df = (
            df.sample(frac=cfg.sample_ratio, random_state=cfg.seed)
              .sort_index()
              .reset_index(drop=True)
        )
    df = add_classification_target(
        df, column="ul_bler", threshold=cfg.threshold, target_name="y_sla_next",
    )
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    feat_cols = schema.categorical + schema.continuous
    split = ood_split_by_tr(df, cfg.train_tr, cfg.val_tr, cfg.test_tr)
    phase_timings["1_parquet_target_split"] = time.time() - t_phase

    t_phase = time.time()
    client_dfs = _partition(split.train, cfg)
    phase_timings["2_partition"] = time.time() - t_phase

    t_phase = time.time()
    client_items = list(client_dfs.items())

    # Per-client sequence build in parallel.
    # 2026-04-26 perf round measurements (Dirichlet α=0.5, 10% data):
    #   - threading n=5: 50.65s (baseline)
    #   - threading n=7: 69.78s (GIL contention with oversubscription)
    #   - loky n=7:      ?      (process-based; no GIL but ~3s startup
    #                            cost per worker for pandas import)
    # Try loky to bypass GIL — biggest client (~1.5× average at α=0.5)
    # otherwise dominates threading wallclock.
    n_workers = min(len(client_items), 7) or 1
    per_client_results = Parallel(n_jobs=n_workers, backend="loky")(
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
    phase_timings["3_per_client_sequences"] = time.time() - t_phase

    if not client_shards:
        raise RuntimeError(
            f"no non-empty clients (alpha={cfg.alpha}, n_clients={cfg.n_clients})"
        )
    log.info(
        "v7 prep: arch=%s algo=%s partition=%s alpha=%.2f n_clients=%d  rows/client=%s",
        cfg.arch, cfg.algorithm, cfg.partition_mode, cfg.alpha,
        len(client_shards),
        {c: len(x) for c, (x, _) in client_shards.items()},
    )

    t_phase = time.time()
    X_va, Y_va = build_run_sequences(
        split.val, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len,
    )
    X_te, Y_te = build_run_sequences(
        split.test, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len,
    )
    phase_timings["4_val_test_sequences"] = time.time() - t_phase

    # Federated scaler (sufficient-stats aggregation, GIL-free joblib).
    t_phase = time.time()
    scaler = federated_fit_scaler(
        {cid: X for cid, (X, _) in client_shards.items()},
        schema,
        n_jobs=len(client_shards),
    )
    phase_timings["5_federated_scaler_fit"] = time.time() - t_phase

    # M5-style pin_memory: mirror fl_v5's ``_maybe_pin``. Pinned host
    # memory enables faster (and truly non-blocking) CPU→GPU transfers
    # via cudaMemcpyAsync during ``.to(device, non_blocking=True)``.
    # Skip on CPU (no transfer) or when sample_ratio is small (pinned
    # tensors live in non-pageable RAM; large allocations stress the OS
    # — only pay this cost when GPU is the target).
    _pin = (device.type == "cuda")

    def _to_tensors(X: np.ndarray, Y: np.ndarray):
        cat, cont = apply_continuous_scaler(X, schema, scaler)
        t_cat = torch.from_numpy(cat)
        t_cont = torch.from_numpy(cont)
        t_y = torch.from_numpy(Y)
        if _pin:
            t_cat = t_cat.pin_memory()
            t_cont = t_cont.pin_memory()
            t_y = t_y.pin_memory()
        return (t_cat, t_cont, t_y)

    t_phase = time.time()
    va_cat, va_cont, va_y = _to_tensors(X_va, Y_va)
    te_cat, te_cont, te_y = _to_tensors(X_te, Y_te)
    client_cpu: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
        cid: _to_tensors(X, Y) for cid, (X, Y) in client_shards.items()
    }
    phase_timings["6_scale_AND_pin_tensors"] = time.time() - t_phase

    # pos_weight from requested split (D-12 contract).
    if cfg.pos_weight_split == "train":
        train_pos = sum(int((Y > 0.5).sum()) for _, Y in client_shards.values())
        train_n = sum(int(len(Y)) for _, Y in client_shards.values())
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
    log.info("v7 prep complete: %.1fs  pos_rate=%.4f  pos_weight=%.4f",
             time.time() - t0, pos_rate, float(pos_weight))

    # ---- Build model ----
    compile_mode = _select_compile_mode(cfg)

    def build_model() -> nn.Module:
        m = _build_model(cfg, schema).to(device)
        if compile_mode and device.type == "cuda":
            m = torch.compile(m, mode=compile_mode)
        return m

    global_model = build_model()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Algorithm instantiation (already validated by _select_algorithm).
    algo_kwargs: dict[str, Any] = {
        "max_steps": cfg.max_steps_per_round,
        "batch_size": cfg.batch_size,
        "grad_clip": cfg.grad_clip,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
    }
    algo_kwargs.update(cfg.algo_kwargs)
    algo_inst = algo_cls(**algo_kwargs)

    # ---- Training rounds ----
    t_phase = time.time()
    cids = sorted(client_cpu.keys())
    rng = np.random.default_rng(cfg.seed)
    history: list[dict] = []
    best_val_auc: float = float("-inf")
    best_state: dict | None = None
    # Record first-round vs steady-state timing separately to detect
    # cold compile / cudnn warmup overhead (per perf-checklist diagnostic).
    first_round_dt: float | None = None
    steady_round_dts: list[float] = []

    for r in range(1, cfg.num_rounds + 1):
        t_round = time.time()
        k = min(cfg.clients_per_round, len(cids))
        selected = rng.choice(cids, size=k, replace=False).tolist()
        global_state = {
            kk: v.detach().clone()
            for kk, v in global_model.state_dict().items()
        }
        lr_this = cfg.lr * min(1.0, r / max(cfg.lr_warmup_rounds, 1))

        updates = []
        for cid in selected:
            local_model = build_model()
            local_model.load_state_dict(global_state, strict=True)
            update = algo_inst.client_update(
                client_id=int(cid),
                local_model=local_model,
                client_tensors=client_cpu[cid],
                loss_fn=loss_fn,
                current_lr=lr_this,
                device=device,
                round_idx=r,
            )
            updates.append(update)
            del local_model

        new_state = algo_inst.server_aggregate(
            global_state=global_state, updates=updates,
        )
        global_model.load_state_dict(new_state)

        val_logits = _batched_predict(global_model, va_cat, va_cont, device)
        val_m = _metrics(va_y[:, 0].numpy().astype(int), val_logits[:, 0])
        train_l = float(np.mean([u.train_loss for u in updates]))
        dt = time.time() - t_round
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
            best_state = {
                kk: v.detach().cpu()
                for kk, v in global_model.state_dict().items()
            }
        if first_round_dt is None:
            first_round_dt = dt
        else:
            steady_round_dts.append(dt)

    phase_timings["7_training_total"] = time.time() - t_phase
    if first_round_dt is not None:
        phase_timings["7a_first_round"] = first_round_dt
    if steady_round_dts:
        phase_timings["7b_steady_round_mean"] = (
            sum(steady_round_dts) / len(steady_round_dts)
        )

    # ---- Test eval on best-val-AUC checkpoint ----
    t_phase = time.time()
    if best_state is not None:
        global_model.load_state_dict(best_state)
    test_logits = _batched_predict(global_model, te_cat, te_cont, device)
    test_m = _metrics(te_y[:, 0].numpy().astype(int), test_logits[:, 0])
    phase_timings["8_test_eval"] = time.time() - t_phase
    phase_timings["TOTAL"] = time.time() - t0
    # Emit phase summary as a single INFO line so post-sweep analysis
    # can grep / parse it from logs without re-running the cell.
    log.info(
        "v7 phase timings (s): %s",
        " | ".join(f"{k}={v:.2f}" for k, v in phase_timings.items()),
    )

    # ---- Emit artifacts (FLAT layout per ADR D-7) ----
    env_meta = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    result = {
        "config": cfg.to_dict(),
        "env": env_meta,
        "history": history,
        "best_val_auc": best_val_auc,
        "test": test_m,
        # Convenience top-level key the aggregator may read directly.
        "test_auc": test_m.get("auc", 0.0),
        # Phase timings (added 2026-04-26 perf-checklist) — keys
        # 1_..8_TOTAL plus 7a/7b first-vs-steady round breakdown.
        "phase_timings_s": phase_timings,
    }
    cell_dir = cfg.output_dir / cfg.name
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    pd.DataFrame(history).to_csv(cell_dir / "history.csv", index=False)
    if best_state is not None:
        torch.save(best_state, cell_dir / "best.pt")

    log.info(
        "%s done: best_val_auc=%.4f  test_auc=%.4f  test_acc=%.4f  test_f1=%.4f",
        cfg.name, best_val_auc,
        test_m.get("auc", 0.0), test_m["accuracy"], test_m["f1"],
    )
    return result
