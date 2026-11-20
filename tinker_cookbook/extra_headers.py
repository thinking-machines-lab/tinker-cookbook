"""Send extra HTTP headers on every Tinker request, configured from the environment.

Some Tinker-compatible servers sit behind a gateway that needs its own auth
headers on top of the Tinker API key. Set ``$TINKER_COOKBOOK_EXTRA_HEADERS`` to a
JSON object of header names to values, and every ``tinker.ServiceClient`` created
after ``import tinker_cookbook`` sends them::

    export TINKER_COOKBOOK_EXTRA_HEADERS='{"X-Gateway-Key": "..."}'

Headers passed explicitly through ``default_headers_overrides`` take precedence.
"""

import functools
import json
import os
from collections.abc import Mapping
from typing import Any

from tinker_cookbook.exceptions import ConfigurationError

EXTRA_HEADERS_ENV_VAR = "TINKER_COOKBOOK_EXTRA_HEADERS"
_INSTALLED_MARKER = "_tinker_cookbook_extra_headers_installed"


def extra_headers_from_env() -> dict[str, str]:
    """Return the headers in ``$TINKER_COOKBOOK_EXTRA_HEADERS``, or ``{}`` when unset."""
    raw = os.environ.get(EXTRA_HEADERS_ENV_VAR)
    if not raw:
        return {}
    try:
        headers = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"{EXTRA_HEADERS_ENV_VAR} is not valid JSON: {e}") from e
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise ConfigurationError(
            f"{EXTRA_HEADERS_ENV_VAR} must be a JSON object of string header names to string values"
        )
    return headers


def install_extra_headers() -> None:
    """Make every ``tinker.ServiceClient`` send the headers from the environment.

    Does nothing when the variable is unset or the hook is already installed.
    """
    headers = extra_headers_from_env()
    if not headers:
        return
    # Imported here so `import tinker_cookbook` stays light when the variable is unset.
    import tinker

    if getattr(tinker.ServiceClient, _INSTALLED_MARKER, False):
        return
    original_init = tinker.ServiceClient.__init__

    @functools.wraps(original_init)
    def init_with_extra_headers(
        self: tinker.ServiceClient,
        *args: Any,
        default_headers_overrides: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        merged = headers | dict(default_headers_overrides or {})
        original_init(self, *args, default_headers_overrides=merged, **kwargs)

    tinker.ServiceClient.__init__ = init_with_extra_headers
    setattr(tinker.ServiceClient, _INSTALLED_MARKER, True)
