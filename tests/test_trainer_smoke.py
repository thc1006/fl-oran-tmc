"""Full FL loop on synthetic data — the definitive 'does it all fit together' test."""
from __future__ import annotations

from pathlib import Path

import pytest

from fl_oran.config import DataConfig, ExperimentConfig, FedConfig, TrainingConfig
from fl_oran.training import run_experiment


@pytest.mark.parametrize("variant", ["v106", "v107_2"])
def test_end_to_end_mlp_variants(variant, synthetic_parquet: Path, tmp_path: Path):
    cfg = ExperimentConfig(
        name=f"smoke_{variant}",
        variant=variant,
        fed=FedConfig(
            num_total_clients=4,
            num_rounds=2,
            clients_per_round=2,
            local_epochs=1,
            batch_size=64,
            early_stopping_patience=0,  # disable
        ),
        data=DataConfig(
            parquet_path=synthetic_parquet,
            samples_per_client=500,
            sample_ratio=None,
            sequence_length=1,
        ),
        training=TrainingConfig(
            device="cpu",
            mixed_precision="off",
            compile_model=False,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=False,
        ),
        output_dir=tmp_path,
    )
    hist = run_experiment(cfg)
    assert len(hist) == 2
    assert "train_loss" in hist.columns
    assert "test_loss" in hist.columns
    assert (tmp_path / "logs").exists()
    assert (tmp_path / "models").exists()


def test_end_to_end_v107_1_lstm(synthetic_parquet: Path, tmp_path: Path):
    cfg = ExperimentConfig(
        name="smoke_v107_1",
        variant="v107_1",
        fed=FedConfig(
            num_total_clients=4,
            num_rounds=2,
            clients_per_round=2,
            local_epochs=1,
            batch_size=32,
            early_stopping_patience=0,
        ),
        data=DataConfig(
            parquet_path=synthetic_parquet,
            samples_per_client=500,
            sample_ratio=None,
            sequence_length=5,
        ),
        training=TrainingConfig(
            device="cpu",
            mixed_precision="off",
            compile_model=False,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=False,
        ),
        output_dir=tmp_path,
    )
    hist = run_experiment(cfg)
    assert len(hist) == 2
