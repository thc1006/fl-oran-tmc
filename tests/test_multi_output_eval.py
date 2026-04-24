"""Multi-output evaluation paths."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_oran.evaluation import binary_classification_metrics, evaluate_multi_output
from fl_oran.models import LSTMMultiOutput, MultiOutputSpec


def test_binary_classification_threshold():
    logits = torch.tensor([[-1.0], [0.5], [2.0]])
    labels = torch.tensor([[0.0], [1.0], [1.0]])
    acc = binary_classification_metrics(logits, labels)
    assert acc == 1.0


def test_evaluate_multi_output_end_to_end():
    torch.manual_seed(0)
    spec = MultiOutputSpec(regression_targets=["a", "b"], classification_targets=["c"])
    model = LSTMMultiOutput(in_features=4, sequence_length=3, output_spec=spec)
    n = 32
    X = torch.randn(n, 3, 4)
    Y = torch.randn(n, spec.n_total)
    Y[:, -1] = (Y[:, -1] > 0).float()  # binary label
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=8, num_workers=0)
    metrics = evaluate_multi_output(
        model, loader, torch.device("cpu"),
        regression_slice=slice(0, 2),
        classification_slice=slice(2, 3),
        amp_enabled=False,
    )
    assert "reg_mse" in metrics
    assert "reg_mae" in metrics
    assert "cls_accuracy" in metrics
    assert 0.0 <= metrics["cls_accuracy"] <= 1.0
