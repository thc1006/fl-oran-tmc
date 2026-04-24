"""Federated trainer orchestrating the three notebook variants."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import ExperimentConfig
from ..data import (
    add_trend_features,
    build_temporal_sequences,
    fit_per_client_scalers,
    load_parquet,
    split_by_client,
)
from ..data.quality import check_quality
from ..evaluation import evaluate_multi_output, evaluate_regression
from ..federated import (
    ClientUpdate,
    GaussianMechanism,
    PrivacyAccountant,
    train_one_client,
    train_one_client_cuda_graph,
    train_one_client_gpu_resident,
    weighted_average_state_dicts,
)
from ..logging_utils import get_logger
from ..models import MultiOutputSpec, build_model
from ..utils import autocast_dtype, log_cuda_info, pick_device, seed_everything

log = get_logger(__name__)


# =============================================================================
# Data pipeline per variant
# =============================================================================

def _prepare_v106_data(cfg: ExperimentConfig) -> tuple[dict[int, tuple], list[str], str]:
    """Load, clean, scale, and return {cid: (X_train, y_train, X_test, y_test)}."""
    df = load_parquet(cfg.data.parquet_path, sample_ratio=cfg.data.sample_ratio, random_state=cfg.fed.random_state)
    features = cfg.get_features()
    target = "allocation_efficiency"
    check_quality(df, features, target, client_col="bs_id")

    clients = split_by_client(
        df,
        samples_per_client=cfg.data.samples_per_client,
        target_column=target,
        preserve_distribution=cfg.data.preserve_distribution,
        random_state=cfg.fed.random_state,
    )
    scalers = fit_per_client_scalers(clients, features, target)

    out: dict[int, tuple] = {}
    for cid, cdf in clients.items():
        X = cdf[features].to_numpy(dtype=np.float32)
        y = cdf[[target]].to_numpy(dtype=np.float32)
        X = scalers.feature_scalers[cid].transform(X).astype(np.float32)
        y = scalers.target_scalers[cid].transform(y).astype(np.float32)
        n_train = int(len(X) * cfg.fed.train_test_split)
        rng = np.random.default_rng(cfg.fed.random_state + cid)
        idx = rng.permutation(len(X))
        tr, te = idx[:n_train], idx[n_train:]
        out[cid] = (X[tr], y[tr], X[te], y[te])
        log.info("client %s: train=%d test=%d", cid, len(tr), len(te))
    return out, features, target


def _prepare_v107_1_data(cfg: ExperimentConfig) -> tuple[dict[int, tuple], list[str], MultiOutputSpec]:
    """Load, add trend features, build temporal sequences, and return per-client tensors."""
    base_cols = [
        "num_ues", "slice_id", "sched_policy_num", "sum_requested_prbs", "network_load",
        "hour", "minute", "day_of_week", "bs_id", "qos_score", "throughput_efficiency", "prb_utilization",
    ]
    df = load_parquet(cfg.data.parquet_path, columns=base_cols, sample_ratio=cfg.data.sample_ratio, random_state=cfg.fed.random_state)

    # Derive a QoS-based SLA-violation label (matches the notebook's fallback when no latency column exists).
    qos_threshold = 0.7
    df["sla_violation"] = (df["qos_score"] < qos_threshold).astype(np.float32)

    df = add_trend_features(df)

    features = cfg.get_features()
    spec = MultiOutputSpec(
        regression_targets=["throughput_efficiency", "qos_score", "prb_utilization"],
        classification_targets=["sla_violation"],
    )

    clients = split_by_client(
        df,
        samples_per_client=cfg.data.samples_per_client,
        target_column="qos_score",
        preserve_distribution=cfg.data.preserve_distribution,
        random_state=cfg.fed.random_state,
    )

    # Feature-scale only (targets stay in original range).
    scalers = fit_per_client_scalers(clients, features, "qos_score")

    out: dict[int, tuple] = {}
    for cid, cdf in clients.items():
        cdf = cdf.sort_index().reset_index(drop=True)
        # Build a float32 feature matrix directly (pandas 3.0 disallows lossy
        # assignment from float32 into uint8/int32 columns).
        X_scaled = scalers.feature_scalers[cid].transform(
            cdf[features].to_numpy(dtype=np.float32)
        ).astype(np.float32)
        cdf_scaled = cdf.drop(columns=features).reset_index(drop=True)
        for i, col in enumerate(features):
            cdf_scaled[col] = X_scaled[:, i]
        cdf = cdf_scaled[features + spec.all_targets]
        seqs, labels = build_temporal_sequences(
            cdf, features, spec.all_targets, seq_len=cfg.data.sequence_length
        )
        if len(seqs) == 0:
            log.warning("client %s had too few rows for seq_len=%d", cid, cfg.data.sequence_length)
            continue
        y_stack = np.hstack([labels[t] for t in spec.all_targets]).astype(np.float32)
        n_train = int(len(seqs) * cfg.fed.train_test_split)
        out[cid] = (
            seqs[:n_train], y_stack[:n_train],
            seqs[n_train:], y_stack[n_train:],
        )
        log.info("client %s: train=%d test=%d (seq)", cid, n_train, len(seqs) - n_train)
    return out, features, spec


# =============================================================================
# Trainer
# =============================================================================

@dataclass
class RoundResult:
    round: int
    train_loss: float
    test_loss: float
    test_metric: float | None = None
    epsilon: float | None = None
    clients: list[int] = field(default_factory=list)
    duration_s: float = 0.0
    extras: dict = field(default_factory=dict)


class FederatedTrainer:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        seed_everything(cfg.fed.random_state, deterministic=cfg.training.deterministic)
        self.device = pick_device(cfg.training.device)
        log_cuda_info(self.device)
        self.amp_enabled, self.amp_dtype = autocast_dtype(cfg.training.mixed_precision)
        log.info("AMP: enabled=%s dtype=%s", self.amp_enabled, self.amp_dtype)

        self.output_dir = Path(cfg.output_dir)
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)

        self.client_data: dict[int, tuple] = {}
        self.features: list[str] = []
        self.multi_output_spec: MultiOutputSpec | None = None
        self.target: str | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        log.info("Preparing data for variant=%s", self.cfg.variant)
        if self.cfg.variant == "v106":
            self.client_data, self.features, self.target = _prepare_v106_data(self.cfg)
        elif self.cfg.variant == "v107_2":
            self.client_data, self.features, self.target = _prepare_v106_data(self.cfg)
        elif self.cfg.variant == "v107_1":
            self.client_data, self.features, self.multi_output_spec = _prepare_v107_1_data(self.cfg)
        else:
            raise ValueError(self.cfg.variant)
        if not self.client_data:
            raise RuntimeError("no client data available")

    def _build_model(self) -> nn.Module:
        kwargs: dict = {}
        if self.cfg.variant == "v107_1":
            assert self.multi_output_spec is not None
            kwargs["output_spec"] = self.multi_output_spec
            kwargs["sequence_length"] = self.cfg.data.sequence_length
        return build_model(self.cfg.variant, in_features=len(self.features), **kwargs)

    def _build_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        nw = self.cfg.training.num_workers
        kwargs: dict = dict(
            dataset=ds,
            batch_size=self.cfg.fed.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=self.cfg.training.pin_memory and self.device.type == "cuda",
            drop_last=False,
        )
        if nw > 0:
            kwargs["prefetch_factor"] = self.cfg.training.prefetch_factor
            kwargs["persistent_workers"] = self.cfg.training.persistent_workers
        return DataLoader(**kwargs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _make_loss_fn(self) -> Callable:
        if self.cfg.variant != "v107_1":
            return nn.MSELoss()
        assert self.multi_output_spec is not None
        n_reg = len(self.multi_output_spec.regression_targets)
        n_cls = len(self.multi_output_spec.classification_targets)
        reg_slice = slice(0, n_reg)
        cls_slice = slice(n_reg, n_reg + n_cls)
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        def combined(pred: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
            reg_loss = mse(pred[:, reg_slice], yb[:, reg_slice]) if n_reg else 0.0
            cls_loss = bce(pred[:, cls_slice], yb[:, cls_slice]) if n_cls else 0.0
            return reg_loss + cls_loss

        return combined

    def _evaluate(self, model: nn.Module, loaders: list[DataLoader]) -> tuple[float, float | None]:
        """Evaluate on the concatenation of all provided test loaders."""
        if self.cfg.variant != "v107_1":
            mses = []
            weights = []
            for ld in loaders:
                m = evaluate_regression(model, ld, self.device, amp_enabled=self.amp_enabled, amp_dtype=self.amp_dtype)
                mses.append(m.mse)
                weights.append(sum(y.numel() for _, y in ld))
            w = np.array(weights, dtype=np.float64)
            w = w / max(w.sum(), 1.0)
            mse = float(np.sum(np.array(mses) * w))
            return mse, None

        assert self.multi_output_spec is not None
        n_reg = len(self.multi_output_spec.regression_targets)
        n_cls = len(self.multi_output_spec.classification_targets)
        reg_slice = slice(0, n_reg)
        cls_slice = slice(n_reg, n_reg + n_cls)
        sum_stats = {"reg_mse": 0.0, "cls_correct": 0, "reg_n": 0, "cls_n": 0}
        for ld in loaders:
            m = evaluate_multi_output(
                model, ld, self.device, reg_slice, cls_slice,
                amp_enabled=self.amp_enabled, amp_dtype=self.amp_dtype,
            )
            # Re-accumulate with per-loader weights.
        # Simpler approach: build a single combined loader via concatenation externally.
        metrics = evaluate_multi_output(
            model, loaders[0], self.device, reg_slice, cls_slice,
            amp_enabled=self.amp_enabled, amp_dtype=self.amp_dtype,
        ) if loaders else {}
        return metrics.get("reg_mse", 0.0), metrics.get("cls_accuracy")

    def run(self) -> pd.DataFrame:
        """Main FL loop."""
        self.prepare_data()
        cids = sorted(self.client_data.keys())

        # --- Fast path: when running on CUDA, park every client's tensors in
        # VRAM once. For 7 clients × 160k × 13 × fp32 this is ~58 MB — trivial
        # for a 16 GB card but eliminates the dominant H2D / DataLoader overhead
        # that otherwise makes batch=64 on a tiny MLP run at ~14% GPU util.
        self._use_gpu_resident = (
            self.device.type == "cuda"
            and self.cfg.training.mixed_precision != "off"  # proxy for "intended to be fast"
        )
        # Pre-build data loaders per client (still needed for evaluation).
        train_loaders = {}
        test_loaders = {}
        gpu_tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for cid in cids:
            Xtr, ytr, Xte, yte = self.client_data[cid]
            train_loaders[cid] = self._build_loader(Xtr, ytr, shuffle=True)
            test_loaders[cid] = self._build_loader(Xte, yte, shuffle=False)
            if self._use_gpu_resident:
                gpu_tensors[cid] = (
                    torch.from_numpy(Xtr).to(self.device, non_blocking=True),
                    torch.from_numpy(ytr).to(self.device, non_blocking=True),
                )
        if self._use_gpu_resident:
            used_mb = sum(x.numel() * x.element_size() + y.numel() * y.element_size()
                          for x, y in gpu_tensors.values()) / (1024 * 1024)
            log.info("GPU-resident training enabled: %.1f MiB of client tensors live on %s", used_mb, self.device)

        # Concatenated test loader for global evaluation.
        Xte_all = np.concatenate([self.client_data[c][2] for c in cids])
        yte_all = np.concatenate([self.client_data[c][3] for c in cids])
        global_test_loader = self._build_loader(Xte_all, yte_all, shuffle=False)

        # Build a single global model — same architecture copied to every client round.
        global_model = self._build_model().to(self.device)
        log.info("Model:\n%s", global_model)

        try:
            if self.cfg.training.compile_model and self.device.type == "cuda":
                global_model = torch.compile(global_model, mode="reduce-overhead", fullgraph=False)
                log.info("torch.compile enabled (reduce-overhead)")
        except Exception as e:  # pragma: no cover
            log.warning("torch.compile failed (%s); proceeding uncompiled", e)

        loss_fn = self._make_loss_fn()

        # DP setup.
        dp = self.cfg.dp
        accountant = None
        if dp.enabled:
            accountant = PrivacyAccountant(
                noise_multiplier=dp.noise_multiplier,
                sample_rate=self.cfg.fed.clients_per_round / self.cfg.fed.num_total_clients,
                target_delta=dp.target_delta,
            )

        history: list[RoundResult] = []
        best_test, best_state, stale = float("inf"), None, 0
        rng = np.random.default_rng(self.cfg.fed.random_state)

        for r in range(1, self.cfg.fed.num_rounds + 1):
            round_start = time.time()
            if dp.enabled and accountant and accountant.epsilon >= dp.target_epsilon:
                log.warning("Privacy budget exhausted: ε=%.3f ≥ %.3f — stopping.", accountant.epsilon, dp.target_epsilon)
                break

            k = min(self.cfg.fed.clients_per_round, len(cids))
            selected = rng.choice(cids, size=k, replace=False).tolist()
            log.info("Round %d/%d | selected clients=%s", r, self.cfg.fed.num_rounds, selected)

            global_state = copy.deepcopy({k: v.detach().cpu() for k, v in global_model.state_dict().items()})
            updates: list[ClientUpdate] = []
            for cid in selected:
                # Spawn a fresh model (architecturally identical) with the current global weights.
                local_model = self._build_model()
                local_model.load_state_dict(global_state, strict=True)
                if self._use_gpu_resident:
                    Xg, yg = gpu_tensors[cid]
                    # Try CUDA graph for every variant; fall back to the plain
                    # GPU-resident path on any capture error (e.g. unsupported ops).
                    trainer_fn = train_one_client_cuda_graph
                    try:
                        update = trainer_fn(
                            cid, local_model, Xg, yg, loss_fn, self.device,
                            lr=self.cfg.fed.client_lr,
                            local_epochs=self.cfg.fed.local_epochs,
                            batch_size=self.cfg.fed.batch_size,
                            amp_enabled=self.amp_enabled,
                            amp_dtype=self.amp_dtype,
                            seed=self.cfg.fed.random_state + r * 1000 + cid,
                        )
                    except RuntimeError as e:  # CUDA-graph capture failed — fall back.
                        log.warning("CUDA graph capture failed for client %s (%s); falling back to GPU-resident path.", cid, e)
                        local_model = self._build_model().to(self.device)
                        local_model.load_state_dict(global_state, strict=True)
                        update = train_one_client_gpu_resident(
                            cid, local_model, Xg, yg, loss_fn, self.device,
                            lr=self.cfg.fed.client_lr,
                            local_epochs=self.cfg.fed.local_epochs,
                            batch_size=self.cfg.fed.batch_size,
                            amp_enabled=self.amp_enabled,
                            amp_dtype=self.amp_dtype,
                            seed=self.cfg.fed.random_state + r * 1000 + cid,
                        )
                else:
                    update = train_one_client(
                        cid, local_model, train_loaders[cid], loss_fn, self.device,
                        lr=self.cfg.fed.client_lr,
                        local_epochs=self.cfg.fed.local_epochs,
                        amp_enabled=self.amp_enabled,
                        amp_dtype=self.amp_dtype,
                    )
                updates.append(update)

            # FedAvg.
            new_state = weighted_average_state_dicts(
                [u.state_dict for u in updates],
                [u.num_examples for u in updates],
            )

            # DP on the weight delta, if enabled.
            if dp.enabled:
                delta = {k: (new_state[k].float() - global_state[k].float()) for k in global_state}
                mech = GaussianMechanism(l2_norm_clip=dp.l2_norm_clip, noise_multiplier=dp.noise_multiplier)
                noisy_delta = mech.clip_and_noise(delta)
                new_state = {k: (global_state[k] + noisy_delta[k]) if noisy_delta[k].dtype.is_floating_point
                             else new_state[k] for k in global_state}
                assert accountant is not None
                accountant.step(num_steps_this_round=1)

            # Load averaged weights back.
            # Strip the `_orig_mod.` prefix that torch.compile adds, if present.
            target_sd = global_model.state_dict()
            fixed = {}
            prefix = "_orig_mod."
            for k, v in new_state.items():
                if k in target_sd:
                    fixed[k] = v
                elif (prefix + k) in target_sd:
                    fixed[prefix + k] = v
                else:
                    fixed[k] = v
            global_model.load_state_dict(fixed, strict=False)

            # Metrics.
            train_loss = float(np.average([u.train_loss for u in updates], weights=[u.num_examples for u in updates]))
            test_loss, test_metric = self._evaluate(global_model, [global_test_loader])
            rr = RoundResult(
                round=r,
                train_loss=train_loss,
                test_loss=test_loss,
                test_metric=test_metric,
                epsilon=accountant.epsilon if accountant else None,
                clients=selected,
                duration_s=time.time() - round_start,
            )
            log.info(
                "Round %d done | train=%.4f test=%.4f %s | dt=%.1fs | ε=%s",
                r, train_loss, test_loss,
                f"acc={test_metric:.3f}" if test_metric is not None else "",
                rr.duration_s,
                f"{rr.epsilon:.3f}" if rr.epsilon is not None else "-",
            )
            history.append(rr)

            if test_loss + 1e-9 < best_test:
                best_test = test_loss
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in global_model.state_dict().items()})
                stale = 0
            else:
                stale += 1
                if self.cfg.fed.early_stopping_patience and stale >= self.cfg.fed.early_stopping_patience:
                    log.info("Early stopping at round %d (no improvement for %d rounds)", r, stale)
                    break

        # Save final artifacts.
        df_hist = pd.DataFrame([asdict(r) for r in history])
        hist_path = self.output_dir / "logs" / f"{self.cfg.name}_history.csv"
        df_hist.to_csv(hist_path, index=False)
        log.info("Saved training history: %s", hist_path)

        if best_state is not None:
            mdl_path = self.output_dir / "models" / f"{self.cfg.name}_best.pt"
            torch.save(best_state, mdl_path)
            log.info("Saved best model weights: %s", mdl_path)

        cfg_path = self.output_dir / "logs" / f"{self.cfg.name}_config.json"
        with open(cfg_path, "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2, default=str)
        log.info("Saved config: %s", cfg_path)

        return df_hist


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    trainer = FederatedTrainer(cfg)
    return trainer.run()
