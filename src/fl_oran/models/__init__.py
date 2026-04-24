from .mlp import MLPv106
from .mlp_deep import MLPv107_2
from .lstm_multi import LSTMMultiOutput, MultiOutputSpec

__all__ = ["MLPv106", "MLPv107_2", "LSTMMultiOutput", "MultiOutputSpec"]


def build_model(variant: str, in_features: int, **kwargs):
    """Factory: map variant -> model class."""
    variant = variant.lower()
    if variant == "v106":
        return MLPv106(in_features=in_features)
    if variant == "v107_2":
        return MLPv107_2(in_features=in_features)
    if variant == "v107_1":
        spec = kwargs.get("output_spec") or MultiOutputSpec(
            regression_targets=["throughput_efficiency", "qos_score", "prb_utilization"],
            classification_targets=["sla_violation"],
        )
        return LSTMMultiOutput(
            in_features=in_features,
            sequence_length=kwargs.get("sequence_length", 5),
            output_spec=spec,
        )
    raise ValueError(f"Unknown variant: {variant}")
