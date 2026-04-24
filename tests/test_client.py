import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_oran.federated import train_one_client
from fl_oran.models import MLPv106


def test_train_one_client_reduces_loss():
    torch.manual_seed(0)
    np.random.seed(0)
    n, d = 256, 13
    X = torch.randn(n, d)
    y = (X.sum(dim=1, keepdim=True) * 0.1).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=32, num_workers=0)
    model = MLPv106(in_features=d)
    loss_fn = torch.nn.MSELoss()
    # Train for several epochs
    # Capture baseline loss from a *single* forward pass.
    with torch.no_grad():
        baseline = loss_fn(model(X), y).item()

    update = train_one_client(
        client_id=7, model=model, loader=loader, loss_fn=loss_fn,
        device=torch.device("cpu"),
        lr=1e-2, local_epochs=5,
        amp_enabled=False, amp_dtype=None,
    )
    assert update.client_id == 7
    assert update.num_examples == n
    # Must converge below the untrained baseline.
    assert update.train_loss < baseline, (update.train_loss, baseline)
    assert update.train_loss > 0
    assert set(update.state_dict.keys()) == {k for k, _ in model.named_parameters()} | {
        k for k, _ in model.named_buffers()
    }
