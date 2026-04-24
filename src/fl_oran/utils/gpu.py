"""GPU setup, AMP helpers, and diagnostic logging."""
from __future__ import annotations

from typing import Literal

import torch

from ..logging_utils import get_logger

log = get_logger(__name__)


def pick_device(request: Literal["cuda", "cpu", "auto"] = "auto") -> torch.device:
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda:0")
    if request == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def autocast_dtype(mode: Literal["off", "fp16", "bf16"]) -> tuple[bool, torch.dtype | None]:
    """Return (enabled, dtype) for torch.autocast."""
    if mode == "off":
        return False, None
    if mode == "fp16":
        return True, torch.float16
    if mode == "bf16":
        return True, torch.bfloat16
    raise ValueError(f"Unknown AMP mode: {mode}")


def log_cuda_info(device: torch.device) -> None:
    if device.type != "cuda":
        log.info("Running on CPU (no CUDA).")
        return
    idx = device.index or 0
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    log.info("CUDA device %d: %s (capability sm_%d%d, %.1f GiB)", idx, name, cap[0], cap[1], total_gb)
    log.info("CUDA version: %s  cuDNN: %s", torch.version.cuda, torch.backends.cudnn.version())
    # bf16 matmul is faster than fp32 on Ada (sm_89) with negligible accuracy impact
    torch.set_float32_matmul_precision("high")
    log.info("float32 matmul precision set to 'high' (TF32)")
