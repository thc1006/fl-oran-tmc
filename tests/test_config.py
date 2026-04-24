from pathlib import Path

import pytest

from fl_oran.config import ExperimentConfig, FEATURES_V106, FEATURES_V107_1


def test_default_experiment_config():
    cfg = ExperimentConfig()
    assert cfg.variant == "v106"
    assert cfg.fed.num_total_clients == 7
    assert cfg.fed.batch_size > 0
    assert cfg.dp.enabled is False


def test_get_features_by_variant():
    cfg = ExperimentConfig(variant="v106")
    assert cfg.get_features() == FEATURES_V106
    cfg = ExperimentConfig(variant="v107_1")
    assert cfg.get_features() == FEATURES_V107_1


def test_experiment_config_yaml_roundtrip(tmp_path: Path):
    import yaml

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({
        "name": "t",
        "variant": "v107_1",
        "fed": {"num_rounds": 5},
        "data": {"parquet_path": "/tmp/x.parquet", "sequence_length": 7},
    }))
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.name == "t"
    assert cfg.variant == "v107_1"
    assert cfg.fed.num_rounds == 5
    assert cfg.data.sequence_length == 7


def test_experiment_config_to_dict_is_json_safe():
    import json
    cfg = ExperimentConfig()
    d = cfg.to_dict()
    json.dumps(d)  # must not raise
    assert isinstance(d["data"]["parquet_path"], str)
