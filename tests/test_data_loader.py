import pandas as pd
import pytest

from fl_oran.data import load_parquet, split_by_client, stratified_sample


def test_load_parquet_basic(synthetic_parquet):
    df = load_parquet(synthetic_parquet)
    assert "bs_id" in df.columns
    assert len(df) > 0


def test_load_parquet_sample_ratio(synthetic_parquet):
    df = load_parquet(synthetic_parquet, sample_ratio=0.1, random_state=1)
    assert 100 <= len(df) <= 1_000  # roughly 10% of 5k


def test_load_parquet_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_parquet(tmp_path / "nope.parquet")


def test_stratified_sample_preserves_distribution(small_dataframe):
    n = 400
    out = stratified_sample(small_dataframe, "allocation_efficiency", n_samples=n)
    assert len(out) == n
    # Rough distribution match (should span the full range).
    assert out["allocation_efficiency"].min() < small_dataframe["allocation_efficiency"].quantile(0.3)
    assert out["allocation_efficiency"].max() > small_dataframe["allocation_efficiency"].quantile(0.7)


def test_stratified_sample_skips_when_already_small(small_dataframe):
    out = stratified_sample(small_dataframe.head(50), "allocation_efficiency", n_samples=1_000)
    assert len(out) == 50


def test_split_by_client_samples_per_client(small_dataframe):
    parts = split_by_client(
        small_dataframe,
        samples_per_client=200,
        target_column="allocation_efficiency",
    )
    assert set(parts.keys()) == set(small_dataframe["bs_id"].unique().tolist())
    for cid, df in parts.items():
        assert len(df) <= 200
