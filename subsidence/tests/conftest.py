import json
from unittest.mock import patch
import pytest


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


@pytest.fixture
def mock_urlopen():
    """Patch urllib.request.urlopen with a programmable queue of responses.

    Use as:
        mock_urlopen.queue.append(_FakeResponse(b'{"access_token": "x"}'))
    """
    with patch("subsidence.ls_client.urllib.request.urlopen") as m:
        queue = []

        def _next_call(req, timeout=None):
            if not queue:
                raise AssertionError(f"unexpected request: {req.full_url}")
            return queue.pop(0)

        m.side_effect = _next_call
        m.queue = queue
        yield m


@pytest.fixture
def fake_response():
    def _make(payload):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload).encode()
        elif isinstance(payload, str):
            payload = payload.encode()
        return _FakeResponse(payload)
    return _make
