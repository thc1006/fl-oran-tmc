import torch

from fl_oran.federated import GaussianMechanism, PrivacyAccountant, gaussian_dp_update


def test_gaussian_mechanism_clip_only():
    mech = GaussianMechanism(l2_norm_clip=1.0, noise_multiplier=0.0)
    delta = {"w": torch.tensor([3.0, 4.0])}  # norm=5
    out = mech.clip_and_noise(delta)
    # Should be clipped to norm 1.
    assert torch.allclose(torch.linalg.vector_norm(out["w"]), torch.tensor(1.0), atol=1e-5)


def test_gaussian_mechanism_noise_is_bounded():
    torch.manual_seed(0)
    mech = GaussianMechanism(l2_norm_clip=1.0, noise_multiplier=0.5)
    delta = {"w": torch.zeros(1000)}
    out = mech.clip_and_noise(delta)
    # With zero delta, output should be pure noise N(0, 0.5).
    assert 0.3 < float(out["w"].std()) < 0.7


def test_gaussian_dp_update_end_to_end():
    before = {"w": torch.tensor([1.0, 2.0])}
    after = {"w": torch.tensor([1.1, 1.9])}  # tiny delta; no clipping triggered
    out = gaussian_dp_update(before, after, l2_norm_clip=10.0, noise_multiplier=0.0)
    assert torch.allclose(out["w"], after["w"])


def test_privacy_accountant_monotonic_eps():
    acc = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.1, target_delta=1e-5)
    eps_seq = [acc.step() for _ in range(5)]
    # ε should be non-decreasing.
    assert all(b >= a - 1e-9 for a, b in zip(eps_seq, eps_seq[1:]))
    assert acc.epsilon == eps_seq[-1]
