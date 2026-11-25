from subsidence import api_constants as C

def test_dataset_names_present():
    for name in ["GNSS_WRA", "MLCW", "DBM", "GW_10MIN", "RAIN_10MIN"]:
        assert hasattr(C, name)

def test_zhuoshui_zone_id():
    assert C.ZHUOSHUI_ZONE_ID == 50

def test_rainfall_sentinel_threshold():
    # negatives become zero
    assert C.RAINFALL_NEG_REPLACE == 0.0
