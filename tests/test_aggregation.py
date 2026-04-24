import torch

from fl_oran.federated import weighted_average_state_dicts


def test_weighted_average_simple():
    s1 = {"w": torch.tensor([0.0, 2.0]), "b": torch.tensor([10.0])}
    s2 = {"w": torch.tensor([4.0, 2.0]), "b": torch.tensor([0.0])}
    avg = weighted_average_state_dicts([s1, s2], [1, 3])
    # Weighted average 1:3 → 0.25 * s1 + 0.75 * s2
    assert torch.allclose(avg["w"], torch.tensor([3.0, 2.0]))
    assert torch.allclose(avg["b"], torch.tensor([2.5]))


def test_weighted_average_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        weighted_average_state_dicts([], [])


def test_weighted_average_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        weighted_average_state_dicts([{"a": torch.zeros(1)}], [1, 2])


def test_integer_tensors_are_copied_not_averaged():
    # BN's num_batches_tracked is long/int.
    s1 = {"w": torch.tensor([1.0]), "n": torch.tensor([3], dtype=torch.long)}
    s2 = {"w": torch.tensor([5.0]), "n": torch.tensor([7], dtype=torch.long)}
    avg = weighted_average_state_dicts([s1, s2], [1, 1])
    assert torch.allclose(avg["w"], torch.tensor([3.0]))
    assert avg["n"].dtype == torch.long
    assert int(avg["n"].item()) == int(s1["n"].item())  # Copied from first
