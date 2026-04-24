"""Tests for the GPU/AMP helpers that don't actually require a GPU."""
from __future__ import annotations

from unittest import mock

import pytest
import torch

from fl_oran.utils.gpu import log_cuda_info, pick_device


def test_pick_device_cuda_when_unavailable_raises():
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError):
            pick_device("cuda")


def test_pick_device_auto_prefers_cuda_if_available():
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        d = pick_device("auto")
        assert d.type == "cuda"


def test_pick_device_auto_falls_back_to_cpu():
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        d = pick_device("auto")
        assert d.type == "cpu"


def test_log_cuda_info_on_cpu_is_noop():
    # Should log and return without touching CUDA APIs.
    log_cuda_info(torch.device("cpu"))
