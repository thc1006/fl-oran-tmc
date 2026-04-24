"""Dataclass-based experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml

# The 13 "classic" features used in v1.0.6 and v1.0.7-2 single-output runs.
FEATURES_V106 = [
    "num_ues", "slice_id", "sched_policy_num", "allocated_rbgs",
    "sum_requested_prbs", "sum_granted_prbs", "prb_utilization",
    "throughput_efficiency", "qos_score", "network_load",
    "hour", "minute", "day_of_week",
]

# The v1.0.7-2 extended set (adds derived KPIs). Only the base 13 are present in
# the provided parquet, so we only use features that exist.
FEATURES_V107_2 = FEATURES_V106  # Parquet does not have the extended columns; fall back to base.

# v1.0.7-1 uses derived-trend features that we synthesize from the base data.
FEATURES_V107_1_BASE = [
    "num_ues", "slice_id", "sched_policy_num", "sum_requested_prbs", "network_load",
    "hour", "minute", "day_of_week",
]
FEATURES_V107_1_TREND = [
    "req_prbs_last3", "req_prbs_change_rate", "req_prbs_volatility",
    "is_peak_hour", "is_weekend",
]
FEATURES_V107_1 = FEATURES_V107_1_BASE + FEATURES_V107_1_TREND


@dataclass
class FedConfig:
    """Federated-learning hyperparameters."""
    num_total_clients: int = 7
    num_rounds: int = 30
    clients_per_round: int = 5
    local_epochs: int = 3
    client_lr: float = 5e-4
    server_momentum: float = 0.0          # 0 = plain FedAvg; >0 = FedAvgM
    batch_size: int = 256
    train_test_split: float = 0.8
    random_state: int = 42
    early_stopping_patience: int = 8      # 0 disables
    drift_threshold: float = 0.2
    drift_patience: int = 5


@dataclass
class DPConfig:
    """Differential-privacy hyperparameters (Gaussian mechanism)."""
    enabled: bool = False
    l2_norm_clip: float = 1.0
    noise_multiplier: float = 0.1
    target_epsilon: float = 10.0
    target_delta: float = 1e-5


@dataclass
class DataConfig:
    """Data ingestion & preprocessing config."""
    parquet_path: Path = field(default_factory=lambda: Path("data/coloran_processed_features.parquet"))
    samples_per_client: int = 200_000
    sample_ratio: float | None = None     # if set, sub-sample the raw parquet first (for smoke tests)
    preserve_distribution: bool = True    # stratified sampling on target
    sequence_length: int = 5              # only used by temporal model (v107-1)


@dataclass
class TrainingConfig:
    """Runtime / optimisation config."""
    device: Literal["cuda", "cpu", "auto"] = "auto"
    mixed_precision: Literal["off", "fp16", "bf16"] = "bf16"
    compile_model: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    deterministic: bool = False


@dataclass
class ExperimentConfig:
    """Top-level experiment config."""
    name: str = "experiment"
    variant: Literal["v106", "v107_1", "v107_2"] = "v106"
    fed: FedConfig = field(default_factory=FedConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "ExperimentConfig":
        fed = FedConfig(**raw.get("fed", {}))
        dp = DPConfig(**raw.get("dp", {}))
        data_d = dict(raw.get("data", {}))
        if "parquet_path" in data_d:
            data_d["parquet_path"] = Path(data_d["parquet_path"])
        data = DataConfig(**data_d)
        training = TrainingConfig(**raw.get("training", {}))
        out_dir = Path(raw.get("output_dir", "artifacts"))
        return cls(
            name=raw.get("name", "experiment"),
            variant=raw.get("variant", "v106"),
            fed=fed,
            dp=dp,
            data=data,
            training=training,
            output_dir=out_dir,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data"]["parquet_path"] = str(self.data.parquet_path)
        d["output_dir"] = str(self.output_dir)
        return d

    def get_features(self) -> list[str]:
        return {
            "v106": FEATURES_V106,
            "v107_1": FEATURES_V107_1,
            "v107_2": FEATURES_V107_2,
        }[self.variant]
