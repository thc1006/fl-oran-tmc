from fl_oran.config import FEATURES_V106
from fl_oran.data.quality import check_quality


def test_quality_on_good_data(small_dataframe):
    rep = check_quality(small_dataframe, FEATURES_V106, "allocation_efficiency")
    assert rep.consistent
    assert isinstance(rep.client_imbalance, dict)


def test_quality_detects_target_out_of_range(small_dataframe):
    bad = small_dataframe.copy()
    bad.loc[bad.index[:10], "allocation_efficiency"] = 5.0
    rep = check_quality(bad, FEATURES_V106, "allocation_efficiency")
    assert not rep.consistent
    assert any("allocation_efficiency" in x for x in rep.issues)
