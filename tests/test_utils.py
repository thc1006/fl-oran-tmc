import torch

from fl_oran.utils import autocast_dtype, pick_device, seed_everything


def test_seed_everything_reproducible():
    seed_everything(123)
    a = torch.randn(3)
    seed_everything(123)
    b = torch.randn(3)
    assert torch.allclose(a, b)


def test_pick_device_cpu():
    d = pick_device("cpu")
    assert d.type == "cpu"


def test_autocast_dtype_modes():
    en, dt = autocast_dtype("off")
    assert (en, dt) == (False, None)
    en, dt = autocast_dtype("bf16")
    assert en and dt == torch.bfloat16
    en, dt = autocast_dtype("fp16")
    assert en and dt == torch.float16


def test_autocast_dtype_invalid():
    import pytest
    with pytest.raises(ValueError):
        autocast_dtype("weird")  # type: ignore[arg-type]
