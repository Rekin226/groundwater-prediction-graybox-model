import pandas as pd
from subsidence.pairing import pair_subsidence_to_gw


def _gw(df_rows):
    return pd.DataFrame(df_rows, columns=["st_id", "gw_st", "gw_TM_X97", "gw_TM_Y97", "zone"])


def _sub(df_rows):
    return pd.DataFrame(df_rows, columns=["sub_id", "sub_dataset", "X_3826", "Y_3826", "zone"])


def test_co_located_match_takes_priority():
    gw = _gw([("st1", 9200211, 100.0, 200.0, 50), ("st2", 9200222, 9000.0, 9000.0, 50)])
    sub = _sub([("YSLL", "ls-wra-gnss-obs", 100.0, 200.0, 50)])
    out = pair_subsidence_to_gw(sub, gw)
    assert out.iloc[0]["pairing_method"] == "co-located"
    assert out.iloc[0]["gw_st"] == 9200211
    assert out.iloc[0]["distance_m"] == 0.0


def test_nn_within_zone_when_no_co_location():
    gw = _gw([("st1", 9200211, 0.0, 0.0, 50), ("st2", 9200222, 5000.0, 0.0, 50)])
    sub = _sub([("X", "ls-wra-gnss-obs", 1000.0, 0.0, 50)])
    out = pair_subsidence_to_gw(sub, gw, co_located_threshold_m=100.0)
    assert out.iloc[0]["pairing_method"] == "nn-within-zone"
    assert out.iloc[0]["gw_st"] == 9200211
    assert abs(out.iloc[0]["distance_m"] - 1000.0) < 0.01


def test_excludes_other_zone():
    gw = _gw([("st1", 9200211, 1.0, 0.0, 60), ("st2", 9200222, 100.0, 0.0, 50)])
    sub = _sub([("X", "ls-wra-gnss-obs", 1.0, 0.0, 50)])
    out = pair_subsidence_to_gw(sub, gw)
    # nearest in zone 50 is st2 even though st1 is closer
    assert out.iloc[0]["gw_st"] == 9200222
