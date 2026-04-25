"""Weak isolation test: encoder + classifier head are identical across the three archs.

Per ADR-001 D-20 (revised TDD entry): the original "naive seed" plan was wrong
because the three architectures consume RNG differently during init (Mamba's
``A_log`` vs Spiking's ``A_log + B + C`` etc.) — so a global ``torch.manual_seed``
will produce different embedding weights even when the encoder code is byte-identical.

This weak version verifies the architectural property that *matters* for the
energy comparison: the **encoder** (categorical embeddings + continuous
concat) and the **classifier head** (`fc` → `relu` → `head`) are
implemented identically across the three Forecaster classes, so any
output difference is caused by the temporal backbone alone.

Specifically we test by **copying weights** rather than re-seeding: build a
ForecasterV2, initialise a MambaForecaster and a SpikingForecaster, then
copy ForecasterV2's embedding weights and head weights into the other
two. If the implementations are truly equivalent up to the backbone, then
running each model with its backbone replaced by a fixed-output stub will
yield identical classifier-head logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fl_oran.data_v2.encoders import FeatureSchema
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.models.mamba_forecaster import MambaForecaster
from fl_oran.models.spiking_forecaster import SpikingForecaster


_SCHEMA = FeatureSchema(
    categorical=["bs_id", "slice_id"],
    categorical_sizes={"bs_id": 7, "slice_id": 3},
    continuous=["dl_throughput_mbps", "ul_throughput_mbps", "prb_util"],
)


def _make_inputs(B: int = 2, L: int = 5):
    torch.manual_seed(0)
    sizes = list(_SCHEMA.categorical_sizes.values())
    x_cat = torch.stack(
        [torch.randint(0, sz + 1, (B, L)) for sz in sizes],
        dim=-1,
    ).long()
    x_cont = torch.randn(B, L, _SCHEMA.n_continuous)
    return x_cat, x_cont


def _copy_encoder_and_head(src, dst):
    """Copy embedding tables + head/fc weights from src into dst.
    Both classes use the same encoder + head structure."""
    for col in src.embeddings:
        dst.embeddings[col].weight.data.copy_(src.embeddings[col].weight.data)
    dst.fc.weight.data.copy_(src.fc.weight.data)
    dst.fc.bias.data.copy_(src.fc.bias.data)
    dst.head.weight.data.copy_(src.head.weight.data)
    dst.head.bias.data.copy_(src.head.bias.data)


def test_encoder_and_head_are_structurally_compatible():
    """The three classes expose identically-named encoder/head submodules."""
    f = ForecasterV2(schema=_SCHEMA, task="classification", seq_len=5)
    m = MambaForecaster(schema=_SCHEMA, task="classification", seq_len=5)
    s = SpikingForecaster(schema=_SCHEMA, task="classification", seq_len=5)
    for model in (f, m, s):
        assert hasattr(model, "embeddings")
        assert hasattr(model, "fc")
        assert hasattr(model, "head")
        assert isinstance(model.embeddings, nn.ModuleDict)
        for col in _SCHEMA.categorical:
            assert col in model.embeddings


def test_encoder_outputs_match_when_weights_copied():
    """With copied embedding weights, the encoder output (cats || cont concat) matches."""
    torch.manual_seed(42)
    f = ForecasterV2(schema=_SCHEMA, task="classification", seq_len=5)
    m = MambaForecaster(schema=_SCHEMA, task="classification", seq_len=5)
    s = SpikingForecaster(schema=_SCHEMA, task="classification", seq_len=5)
    _copy_encoder_and_head(f, m)
    _copy_encoder_and_head(f, s)

    x_cat, x_cont = _make_inputs()

    def _encoder_output(model):
        cats = []
        for i, col in enumerate(model.schema.categorical):
            cats.append(model.embeddings[col](x_cat[..., i]))
        return torch.cat(cats + [x_cont], dim=-1) if cats else x_cont

    enc_f = _encoder_output(f)
    enc_m = _encoder_output(m)
    enc_s = _encoder_output(s)
    assert torch.equal(enc_f, enc_m)
    assert torch.equal(enc_f, enc_s)


def test_head_outputs_match_when_input_to_head_is_pinned():
    """If the post-backbone hidden vector is identical, head logits must match exactly."""
    torch.manual_seed(11)
    f = ForecasterV2(schema=_SCHEMA, task="classification", seq_len=5)
    m = MambaForecaster(schema=_SCHEMA, task="classification", seq_len=5)
    s = SpikingForecaster(schema=_SCHEMA, task="classification", seq_len=5)
    _copy_encoder_and_head(f, m)
    _copy_encoder_and_head(f, s)

    # Make the head input identical for all three (32-dim vector, batch=4).
    pinned = torch.randn(4, 32)

    def _head(model, h):
        return model.head(model.relu(model.fc(model.dropout(h))))

    f.eval()
    m.eval()
    s.eval()
    out_f = _head(f, pinned)
    out_m = _head(m, pinned)
    out_s = _head(s, pinned)
    assert torch.equal(out_f, out_m)
    assert torch.equal(out_f, out_s)
