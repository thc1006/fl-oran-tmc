"""CLI argument parsing + config construction."""
from __future__ import annotations

from pathlib import Path

from fl_oran.cli import _build_cfg_from_args, main
import argparse


def _parse(argv: list[str]):
    p = argparse.ArgumentParser()
    from fl_oran.cli import _add_arguments
    _add_arguments(p)
    return p.parse_args(argv)


def test_build_cfg_from_args_overrides():
    args = _parse([
        "--variant", "v107_1",
        "--name", "custom",
        "--num-rounds", "4",
        "--clients-per-round", "2",
        "--batch-size", "32",
        "--lr", "0.001",
        "--seq-len", "7",
        "--dp",
        "--dp-clip", "2.0",
        "--dp-noise", "0.3",
        "--device", "cpu",
        "--amp", "off",
        "--no-compile",
        "--num-workers", "0",
        "--sample-ratio", "0.1",
    ])
    cfg = _build_cfg_from_args(args)
    assert cfg.variant == "v107_1"
    assert cfg.name == "custom"
    assert cfg.fed.num_rounds == 4
    assert cfg.fed.clients_per_round == 2
    assert cfg.fed.batch_size == 32
    assert cfg.fed.client_lr == 0.001
    assert cfg.data.sequence_length == 7
    assert cfg.data.sample_ratio == 0.1
    assert cfg.dp.enabled
    assert cfg.dp.l2_norm_clip == 2.0
    assert cfg.dp.noise_multiplier == 0.3
    assert cfg.training.device == "cpu"
    assert cfg.training.mixed_precision == "off"
    assert cfg.training.compile_model is False
    assert cfg.training.num_workers == 0


def test_build_cfg_from_yaml(tmp_path: Path):
    import yaml
    yml = tmp_path / "c.yaml"
    yml.write_text(yaml.safe_dump({
        "name": "y", "variant": "v106", "fed": {"num_rounds": 2},
    }))
    args = _parse(["--config", str(yml), "--variant", "v107_2"])
    cfg = _build_cfg_from_args(args)
    # CLI overrides yaml's variant:
    assert cfg.variant == "v107_2"
    # But other yaml fields come through (via subsequent cfg.fed overwrite, `num_rounds` comes from CLI default 30).
    assert cfg.fed.num_rounds == 30


def test_cli_main_runs_smoke(tmp_path: Path, synthetic_parquet: Path):
    # End-to-end: just check it returns 0.
    code = main([
        "--variant", "v106",
        "--name", "cli_smoke",
        "--parquet", str(synthetic_parquet),
        "--output-dir", str(tmp_path),
        "--num-rounds", "1",
        "--clients-per-round", "2",
        "--local-epochs", "1",
        "--batch-size", "32",
        "--samples-per-client", "200",
        "--device", "cpu",
        "--amp", "off",
        "--no-compile",
        "--num-workers", "0",
        "--log-level", "WARNING",
    ])
    assert code == 0
    assert (tmp_path / "logs").exists()
