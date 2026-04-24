import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_oran.evaluation import evaluate_regression, regression_metrics


def test_regression_metrics_math():
    pred = torch.tensor([[1.0], [2.0], [3.0]])
    true = torch.tensor([[1.0], [2.5], [2.5]])
    m = regression_metrics(pred, true)
    assert abs(m.mse - ((0 + 0.25 + 0.25) / 3)) < 1e-6
    assert abs(m.mae - ((0 + 0.5 + 0.5) / 3)) < 1e-6


def test_evaluate_regression_end_to_end():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    x = torch.randn(64, 4)
    y = model(x).detach() + 0.0  # exact target = zero loss
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=16, num_workers=0)
    m = evaluate_regression(model, loader, torch.device("cpu"), amp_enabled=False)
    assert m.mse < 1e-5
