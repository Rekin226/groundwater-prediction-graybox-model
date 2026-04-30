import pytest
from subsidence.ls_client import to_api_id, from_api_id


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
