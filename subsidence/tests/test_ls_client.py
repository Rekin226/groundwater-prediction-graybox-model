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
