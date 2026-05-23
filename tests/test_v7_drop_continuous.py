"""drop_continuous (no-BLER leakage-control ablation) invariants.

Locks the safety properties reviewed for PR #30: dropping continuous features
reduces schema.n_continuous and the model input_dim by exactly that count, the
dropped names are gone, and no persistence skip is introduced (persistence_feature
is None -> no second leak path). The unknown-feature guard in fl_v7's data prep
(raise on a typo'd name) prevents a silent no-op ablation.
"""
import fl_oran.training.fl_v7 as m
from fl_oran.models.forecaster_v2 import ForecasterV2


def _schema(drop):
    cont = [c for c in m.V3_CONTINUOUS if c not in drop]
    return m.FeatureSchema(
        categorical=m.V3_CATEGORICAL, categorical_sizes=m.V3_CAT_SIZES, continuous=cont
    ), cont


def test_drop_bler_reduces_features_and_model_input_dim():
    s0, _ = _schema([])
    s1, c1 = _schema(["dl_bler", "ul_bler"])
    assert s1.n_continuous == s0.n_continuous - 2
    assert "dl_bler" not in c1 and "ul_bler" not in c1
    m0 = ForecasterV2(schema=s0, task="classification", seq_len=5)
    m1 = ForecasterV2(schema=s1, task="classification", seq_len=5)
    assert m1.lstm1.input_size == m0.lstm1.input_size - 2
    assert m1.persistence_feature is None  # no skip => no second leak path


def test_drop_continuous_field_defaults_empty():
    assert m.V7Config().drop_continuous == []


def test_dropped_bler_are_real_v3_continuous_names():
    # the ablation only means something if the dropped names actually exist
    assert "dl_bler" in m.V3_CONTINUOUS and "ul_bler" in m.V3_CONTINUOUS
