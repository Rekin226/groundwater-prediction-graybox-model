import pytest
import pandas as pd
from subsidence.ls_client import to_api_id, from_api_id, split_to_df


def test_to_api_id_pads_to_eight():
    assert to_api_id(9200211) == "09200211"


def test_to_api_id_already_eight_digits():
    assert to_api_id(99200211) == "99200211"


def test_to_api_id_accepts_string():
    assert to_api_id("9200211") == "09200211"


def test_to_api_id_rejects_too_long():
    with pytest.raises(ValueError):
        to_api_id(123456789)


def test_from_api_id_strips_leading_zero():
    assert from_api_id("09200211") == 9200211


def test_from_api_id_no_leading_zero():
    assert from_api_id("99200211") == 99200211


def test_roundtrip():
    assert from_api_id(to_api_id(9200211)) == 9200211


import os
from subsidence.ls_client import LSClient


def test_get_token_calls_token_url(monkeypatch, mock_urlopen, fake_response):
    monkeypatch.setenv("LS_USER", "u")
    monkeypatch.setenv("LS_PASS", "p")
    mock_urlopen.queue.append(fake_response({"access_token": "TOKEN", "token_type": "bearer"}))
    c = LSClient()
    tok = c.get_token()
    assert tok == "TOKEN"
    # Ensure exactly one call, to /apis/token
    call_req = mock_urlopen.call_args[0][0]
    assert call_req.full_url == "https://api.wisenvr.com/apis/token"


def test_get_json_attaches_bearer(monkeypatch, mock_urlopen, fake_response):
    monkeypatch.setenv("LS_USER", "u")
    monkeypatch.setenv("LS_PASS", "p")
    # token, then data
    mock_urlopen.queue.append(fake_response({"access_token": "TOKEN", "token_type": "bearer"}))
    mock_urlopen.queue.append(fake_response({"foo": 1}))
    c = LSClient()
    res = c.get_json("/something", params={"k": "v"})
    assert res == {"foo": 1}
    # second call should have Authorization header
    second_req = mock_urlopen.call_args_list[1][0][0]
    assert second_req.headers.get("Authorization") == "Bearer TOKEN"
    assert "k=v" in second_req.full_url


def test_missing_env_raises(monkeypatch):
    monkeypatch.delenv("LS_USER", raising=False)
    monkeypatch.delenv("LS_PASS", raising=False)
    with pytest.raises(RuntimeError, match="LS_USER"):
        LSClient().get_token()


from pathlib import Path

def test_cached_get_dataframe_writes_parquet(tmp_path, monkeypatch, mock_urlopen, fake_response):
    monkeypatch.setenv("LS_USER", "u"); monkeypatch.setenv("LS_PASS", "p")
    # Token then data
    mock_urlopen.queue.append(fake_response({"access_token": "T", "token_type": "bearer"}))
    mock_urlopen.queue.append(fake_response({
        "columns": ["value"],
        "index": ["2020-01-01T00:00:00", "2020-01-02T00:00:00"],
        "data": [[1.0], [2.0]],
    }))
    c = LSClient()
    df = c.cached_get_dataframe(
        path="/dataset/x/station/y/data",
        params={"orient": "split"},
        cache_dir=tmp_path, cache_key="x__y",
    )
    assert list(df.columns) == ["value"]
    assert df.iloc[0, 0] == 1.0
    parquets = list(tmp_path.glob("*.parquet"))
    assert len(parquets) == 1


def test_cached_get_dataframe_uses_cache_on_second_call(tmp_path, monkeypatch, mock_urlopen, fake_response):
    monkeypatch.setenv("LS_USER", "u"); monkeypatch.setenv("LS_PASS", "p")
    mock_urlopen.queue.append(fake_response({"access_token": "T", "token_type": "bearer"}))
    mock_urlopen.queue.append(fake_response({
        "columns": ["value"], "index": ["2020-01-01T00:00:00"], "data": [[42.0]],
    }))
    c = LSClient()
    df1 = c.cached_get_dataframe("/p", cache_dir=tmp_path, cache_key="k")
    # second call should NOT make any HTTP request — queue was consumed
    df2 = c.cached_get_dataframe("/p", cache_dir=tmp_path, cache_key="k")
    pd.testing.assert_frame_equal(df1, df2)
    assert mock_urlopen.call_count == 2  # token + first data, no extra


def test_split_to_df_preserves_non_datetime_index_when_flag_false():
    payload = {
        "index": ["YSLL", "TKJS", "新街國小"],
        "columns": ["lat", "lon"],
        "data": [[23.7, 120.2], [23.6, 120.4], [23.9, 120.3]],
    }
    df = split_to_df(payload, datetime_index=False)
    assert list(df.index) == ["YSLL", "TKJS", "新街國小"]
    assert df.loc["YSLL", "lat"] == 23.7


def test_split_to_df_default_coerces_to_datetime():
    payload = {
        "index": ["2020-01-01T00:00:00", "2020-01-02T00:00:00"],
        "columns": ["value"],
        "data": [[1.0], [2.0]],
    }
    df = split_to_df(payload)  # default datetime_index=True
    assert isinstance(df.index, pd.DatetimeIndex)
