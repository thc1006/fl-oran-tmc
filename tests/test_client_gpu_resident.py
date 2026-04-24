"""Tests for the GPU-resident training path (runs on CPU via device='cpu')."""
from __future__ import annotations

import torch

from fl_oran.federated import train_one_client_gpu_resident
from fl_oran.models import MLPv106


def test_gpu_resident_trains_on_cpu_device():
    torch.manual_seed(0)
    device = torch.device("cpu")
    n, d = 256, 13
    X = torch.randn(n, d, device=device)
    y = (X.sum(dim=1, keepdim=True) * 0.1).float()
    model = MLPv106(in_features=d)
    with torch.no_grad():
        baseline = torch.nn.functional.mse_loss(model(X), y).item()
    update = train_one_client_gpu_resident(
        client_id=3, model=model, X=X, y=y,
        loss_fn=torch.nn.MSELoss(), device=device,
        lr=1e-2, local_epochs=5, batch_size=64,
        amp_enabled=False, amp_dtype=None, seed=0,
    )
    assert update.client_id == 3
    assert update.num_examples == n
    assert update.train_loss < baseline
    assert update.train_loss > 0


def test_gpu_resident_rejects_wrong_device():
    import pytest
    device = torch.device("cpu")
    X = torch.randn(8, 4, device="cpu")
    y = torch.randn(8, 1, device="cpu")
    # Mis-declare device: expect assertion.
    wrong_device = torch.device("meta")
    with pytest.raises(AssertionError):
        train_one_client_gpu_resident(
            client_id=0, model=MLPv106(4), X=X, y=y,
            loss_fn=torch.nn.MSELoss(), device=wrong_device,
            lr=1e-3, local_epochs=1, batch_size=4,
            amp_enabled=False, amp_dtype=None,
        )
