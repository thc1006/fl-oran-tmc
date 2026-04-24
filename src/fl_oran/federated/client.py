"""Per-client local training step."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..logging_utils import get_logger

log = get_logger(__name__)


def _make_optimizer(
    params, lr: float, device: torch.device, *, capturable: bool = False
) -> torch.optim.Optimizer:
    """Fused Adam on CUDA; plain Adam elsewhere.

    ``capturable=True`` is required when the ``.step()`` call must be captured
    into a CUDA graph (see ``train_one_client_cuda_graph``).
    """
    if device.type == "cuda":
        try:
            return torch.optim.Adam(params, lr=lr, fused=True, capturable=capturable)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(params, lr=lr)


@dataclass
class ClientUpdate:
    client_id: int
    state_dict: dict[str, torch.Tensor]
    num_examples: int
    train_loss: float
    train_metric: float | None = None
    # Algorithm-specific auxiliary payload (SCAFFOLD control-variate delta,
    # FedDyn gradient residual, etc.). Empty dict for FedAvg/FedProx/FedAdam.
    aux: dict[str, Any] = field(default_factory=dict)


def train_one_client_cuda_graph(
    client_id: int,
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    device: torch.device,
    *,
    lr: float,
    local_epochs: int,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    seed: int | None = None,
) -> ClientUpdate:
    """Fastest path: captures forward+backward+optimizer.step() into a CUDA
    graph and replays it for every mini-batch.

    Eliminates almost all Python-side kernel-launch overhead. For a tiny
    13→64→32→1 MLP at batch=64 this is the difference between ~15 s/round
    (launch-bound) and ~2 s/round (graph-replay bound).

    Partial trailing batches are dropped (static-shape requirement). At 200 k
    samples with batch=64 we lose at most 63 samples per epoch.
    """
    assert device.type == "cuda", "CUDA graphs require a CUDA device"
    assert X.is_cuda and y.is_cuda
    model.to(device).train()
    optimizer = _make_optimizer(model.parameters(), lr, device, capturable=True)

    n = X.shape[0]
    n_full = (n // batch_size) * batch_size  # drop_last
    if n_full == 0:
        raise ValueError("not enough samples to form a full batch")

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # Static input/output tensors for the captured graph.
    static_x = torch.zeros((batch_size,) + X.shape[1:], device=device, dtype=X.dtype)
    static_y = torch.zeros((batch_size,) + y.shape[1:], device=device, dtype=y.dtype)
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_enabled)

    # ---- Warm-up on a side stream (required before capture) ----
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                pred = model(static_x)
                loss = loss_fn(pred, static_y)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    # ---- Capture the step ----
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        with amp_ctx:
            static_pred = model(static_x)
            static_loss = loss_fn(static_pred, static_y)
        static_loss.backward()
        optimizer.step()

    # ---- Replay ----
    total_loss, seen = 0.0, 0
    for _ in range(local_epochs):
        perm = torch.randperm(n, generator=gen, device=device)[:n_full]
        # pre-gather for the whole epoch to avoid per-batch indexing overhead
        X_perm = X.index_select(0, perm)
        y_perm = y.index_select(0, perm)
        for i in range(0, n_full, batch_size):
            static_x.copy_(X_perm[i:i + batch_size], non_blocking=True)
            static_y.copy_(y_perm[i:i + batch_size], non_blocking=True)
            g.replay()
            total_loss += float(static_loss.detach()) * batch_size
            seen += batch_size

    avg_loss = total_loss / max(seen, 1)
    log.debug("client %s (cuda-graph): seen=%d loss=%.4f", client_id, seen, avg_loss)
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    return ClientUpdate(
        client_id=client_id,
        state_dict=state,
        num_examples=n_full,
        train_loss=avg_loss,
    )


def train_one_client_gpu_resident(
    client_id: int,
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    device: torch.device,
    *,
    lr: float,
    local_epochs: int,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    metric_fn: Callable | None = None,
    seed: int | None = None,
) -> ClientUpdate:
    """Fast path: data already lives on the target device — no DataLoader, no H2D.

    Iterates ``local_epochs`` × ``ceil(n/batch_size)`` batches of pure index slicing.
    Eliminates DataLoader/process overhead, which for tiny MLPs with batch=64
    dominates wall-clock time. Uses fused Adam on CUDA.
    """
    assert X.device == device and y.device == device, (
        f"tensors must already be on {device}; got X on {X.device}, y on {y.device}"
    )
    model.to(device).train()
    optimizer = _make_optimizer(model.parameters(), lr, device)
    n = X.shape[0]
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_enabled
    )
    total_loss, total_metric, seen_samples = 0.0, 0.0, 0
    for _ in range(local_epochs):
        perm = torch.randperm(n, generator=gen, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx]
            yb = y[idx]
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                pred = model(xb)
                loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            b = xb.shape[0]
            total_loss += loss.item() * b
            seen_samples += b
            if metric_fn is not None:
                total_metric += metric_fn(pred.detach(), yb).item() * b

    avg_loss = total_loss / max(seen_samples, 1)
    avg_metric = (total_metric / max(seen_samples, 1)) if metric_fn is not None else None
    log.debug(
        "client %s (gpu-resident): n=%d epochs=%d loss=%.4f", client_id, n, local_epochs, avg_loss
    )
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    return ClientUpdate(
        client_id=client_id,
        state_dict=state,
        num_examples=n,
        train_loss=avg_loss,
        train_metric=avg_metric,
    )


def train_one_client(
    client_id: int,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    *,
    lr: float,
    local_epochs: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    metric_fn: Callable | None = None,
) -> ClientUpdate:
    """Run ``local_epochs`` of SGD over ``loader`` starting from the current model
    state. Returns the resulting state_dict plus metrics."""
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_loss, total_metric, seen_samples, seen_batches = 0.0, 0.0, 0, 0

    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_enabled
    )
    # FedAvg weight = dataset size, not sum-over-epochs.
    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else None
    for _ in range(local_epochs):
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                xb, yb = batch
            else:
                xb, yb = batch["x"], batch["y"]
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                pred = model(xb)
                loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            n = xb.shape[0]
            total_loss += loss.item() * n
            seen_samples += n
            seen_batches += 1
            if metric_fn is not None:
                total_metric += metric_fn(pred.detach(), yb).item() * n

    avg_loss = total_loss / max(seen_samples, 1)
    avg_metric = (total_metric / max(seen_samples, 1)) if metric_fn is not None else None
    n_unique = dataset_size if dataset_size is not None else seen_samples // max(local_epochs, 1)
    log.debug(
        "client %s: n=%d loss=%.4f metric=%s",
        client_id, n_unique, avg_loss, f"{avg_metric:.4f}" if avg_metric else "-",
    )
    # Offload state to CPU so the server side keeps VRAM free.
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    return ClientUpdate(
        client_id=client_id,
        state_dict=state,
        num_examples=n_unique,
        train_loss=avg_loss,
        train_metric=avg_metric,
    )
