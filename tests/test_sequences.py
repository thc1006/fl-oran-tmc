import numpy as np
import pandas as pd

from fl_oran.data import build_temporal_sequences, add_trend_features


def test_build_temporal_sequences_shapes():
    df = pd.DataFrame({
        "f1": np.arange(10, dtype=np.float32),
        "f2": np.arange(10, dtype=np.float32) * 2,
        "target": np.arange(10, dtype=np.float32) + 100,
    })
    seqs, labels = build_temporal_sequences(df, ["f1", "f2"], ["target"], seq_len=3)
    assert seqs.shape == (8, 3, 2)
    assert labels["target"].shape == (8, 1)
    # Label at index i should equal df.target[i+seq_len-1] = df.target[i+2].
    np.testing.assert_allclose(labels["target"][0, 0], df["target"].iloc[2])


def test_build_temporal_sequences_too_short():
    df = pd.DataFrame({"f1": [1.0, 2.0], "target": [10.0, 20.0]})
    seqs, labels = build_temporal_sequences(df, ["f1"], ["target"], seq_len=5)
    assert seqs.shape[0] == 0


def test_add_trend_features_creates_columns():
    df = pd.DataFrame({
        "bs_id": [1] * 8 + [2] * 8,
        "sum_requested_prbs": np.arange(16.0),
        "hour": [6, 7, 19, 20, 8, 9, 21, 23, 1, 2, 18, 22, 12, 13, 14, 15],
        "day_of_week": [0, 1, 5, 6, 2, 3, 4, 5, 0, 6, 3, 4, 1, 5, 2, 0],
    })
    out = add_trend_features(df)
    for col in ["req_prbs_last3", "req_prbs_change_rate", "req_prbs_volatility", "is_peak_hour", "is_weekend"]:
        assert col in out.columns
    # Peak hour 18..22.
    assert int(out.loc[out["hour"] == 20, "is_peak_hour"].iloc[0]) == 1
    assert int(out.loc[out["hour"] == 15, "is_peak_hour"].iloc[0]) == 0
    # Weekend: dow >= 5.
    assert int(out.loc[out["day_of_week"] == 6, "is_weekend"].iloc[0]) == 1
    assert int(out.loc[out["day_of_week"] == 1, "is_weekend"].iloc[0]) == 0
    # Change rate is bounded.
    assert (out["req_prbs_change_rate"].abs() <= 1.0 + 1e-6).all()
