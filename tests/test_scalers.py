import numpy as np

from fl_oran.config import FEATURES_V106
from fl_oran.data import fit_per_client_scalers, split_by_client


def test_per_client_scalers_fit_and_transform(small_dataframe):
    parts = split_by_client(small_dataframe, samples_per_client=500, target_column="allocation_efficiency")
    scalers = fit_per_client_scalers(parts, FEATURES_V106, "allocation_efficiency")

    # Every client must have a fitted scaler.
    assert set(scalers.feature_scalers.keys()) == set(parts.keys())

    # Global scaler means should be a weighted average.
    expected_total = sum(len(df) for df in parts.values())
    assert scalers.global_feature_scaler.n_samples_seen_ == expected_total

    # Transforming with per-client scaler returns ~zero-mean per-client.
    first_cid = next(iter(parts))
    X = parts[first_cid][FEATURES_V106].to_numpy(dtype=np.float32)
    Xz = scalers.feature_scalers[first_cid].transform(X)
    assert abs(float(Xz.mean())) < 0.1


def test_inverse_transform_target_recovers(small_dataframe):
    parts = split_by_client(small_dataframe, samples_per_client=500, target_column="allocation_efficiency")
    scalers = fit_per_client_scalers(parts, FEATURES_V106, "allocation_efficiency")
    cid = next(iter(parts))
    y = parts[cid][["allocation_efficiency"]].to_numpy(dtype=np.float32)
    yz = scalers.target_scalers[cid].transform(y).ravel()
    y_back = scalers.inverse_transform_target(cid, yz)
    np.testing.assert_allclose(y_back, y.ravel(), rtol=1e-3, atol=1e-3)
