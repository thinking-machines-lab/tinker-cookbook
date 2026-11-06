from typing import Any

import pytest
import tinker

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.extra_headers import (
    _INSTALLED_MARKER,
    EXTRA_HEADERS_ENV_VAR,
    extra_headers_from_env,
    install_extra_headers,
)


@pytest.fixture
def init_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ServiceClient.__init__ with a recorder and clear the installed marker."""
    calls: list[dict[str, Any]] = []

    def recording_init(self: tinker.ServiceClient, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(tinker.ServiceClient, "__init__", recording_init)
    monkeypatch.setattr(tinker.ServiceClient, _INSTALLED_MARKER, False, raising=False)
    return calls


def test_headers_sent_by_every_client(
    monkeypatch: pytest.MonkeyPatch, init_calls: list[dict[str, Any]]
):
    monkeypatch.setenv(EXTRA_HEADERS_ENV_VAR, '{"X-Gateway-Key": "k", "X-Gateway-Secret": "s"}')
    install_extra_headers()
    tinker.ServiceClient(base_url="http://example.com")
    assert init_calls == [
        {
            "base_url": "http://example.com",
            "default_headers_overrides": {"X-Gateway-Key": "k", "X-Gateway-Secret": "s"},
        }
    ]


def test_caller_overrides_win(monkeypatch: pytest.MonkeyPatch, init_calls: list[dict[str, Any]]):
    monkeypatch.setenv(EXTRA_HEADERS_ENV_VAR, '{"X-Gateway-Key": "env", "X-Other": "o"}')
    install_extra_headers()
    tinker.ServiceClient(default_headers_overrides={"X-Gateway-Key": "caller"})
    assert init_calls[0]["default_headers_overrides"] == {
        "X-Gateway-Key": "caller",
        "X-Other": "o",
    }


def test_unset_leaves_client_untouched(
    monkeypatch: pytest.MonkeyPatch, init_calls: list[dict[str, Any]]
):
    monkeypatch.delenv(EXTRA_HEADERS_ENV_VAR, raising=False)
    original_init = tinker.ServiceClient.__init__
    install_extra_headers()
    assert tinker.ServiceClient.__init__ is original_init


def test_install_twice_wraps_once(
    monkeypatch: pytest.MonkeyPatch, init_calls: list[dict[str, Any]]
):
    monkeypatch.setenv(EXTRA_HEADERS_ENV_VAR, '{"X-Gateway-Key": "k"}')
    install_extra_headers()
    wrapped_init = tinker.ServiceClient.__init__
    install_extra_headers()
    assert tinker.ServiceClient.__init__ is wrapped_init


@pytest.mark.parametrize("raw", ["not json", '["X-Key", "k"]', '{"X-Key": 1}'])
def test_rejects_bad_values(monkeypatch: pytest.MonkeyPatch, raw: str):
    monkeypatch.setenv(EXTRA_HEADERS_ENV_VAR, raw)
    with pytest.raises(ConfigurationError, match=EXTRA_HEADERS_ENV_VAR):
        extra_headers_from_env()
