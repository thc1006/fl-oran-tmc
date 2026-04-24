import torch

from fl_oran.models import MLPv106, MLPv107_2, LSTMMultiOutput, MultiOutputSpec, build_model


def test_mlp_v106_forward():
    m = MLPv106(in_features=13)
    x = torch.randn(8, 13)
    y = m(x)
    assert y.shape == (8, 1)


def test_mlp_v107_2_forward():
    m = MLPv107_2(in_features=13)
    m.train()
    x = torch.randn(8, 13)
    y = m(x)
    assert y.shape == (8, 1)
    # BN needs batch > 1 in train mode; eval should work on any size.
    m.eval()
    y1 = m(torch.randn(1, 13))
    assert y1.shape == (1, 1)


def test_lstm_multi_output_forward():
    spec = MultiOutputSpec(regression_targets=["a", "b", "c"], classification_targets=["d"])
    m = LSTMMultiOutput(in_features=13, sequence_length=5, output_spec=spec)
    x = torch.randn(8, 5, 13)
    y = m(x)
    assert y.shape == (8, spec.n_total)


def test_build_model_factory():
    assert isinstance(build_model("v106", in_features=13), MLPv106)
    assert isinstance(build_model("v107_2", in_features=13), MLPv107_2)
    assert isinstance(build_model("v107_1", in_features=13, sequence_length=5), LSTMMultiOutput)


def test_build_model_unknown_variant_raises():
    try:
        build_model("nonsense", in_features=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
